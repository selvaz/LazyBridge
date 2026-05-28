"""``lazybridge.matrix`` — declarative provider-capability lookup.

Each provider class declares its capabilities as ``ClassVar`` flags
(see :class:`~lazybridge.core.providers.base.BaseProvider`).  This
module aggregates them into a single typed dict so docs / introspection
tools / capability-aware error messages can read the support matrix
without importing each provider individually.

Usage::

    from lazybridge.matrix import provider_capabilities

    for name, caps in provider_capabilities().items():
        print(name, caps)

The dict is computed lazily on first call and cached, so repeated
imports stay cheap.  Adding a new provider is one line in
:mod:`lazybridge.core.providers` (``register`` it) plus the four
ClassVars on the provider class itself.
"""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from lazybridge.core.types import NativeTool

#: ``provider name → (module, attribute)`` for every provider LLMEngine can
#: instantiate.  Imported lazily and *individually* (see
#: :func:`provider_capabilities`) so a single broken provider SDK degrades
#: that one row instead of blinding the whole introspection matrix.
_PROVIDER_IMPORTS: list[tuple[str, str, str]] = [
    ("anthropic", "lazybridge.core.providers.anthropic", "AnthropicProvider"),
    ("openai", "lazybridge.core.providers.openai", "OpenAIProvider"),
    ("google", "lazybridge.core.providers.google", "GoogleProvider"),
    ("deepseek", "lazybridge.core.providers.deepseek", "DeepSeekProvider"),
    ("litellm", "lazybridge.core.providers.litellm", "LiteLLMProvider"),
    ("lmstudio", "lazybridge.core.providers.lmstudio", "LMStudioProvider"),
]


@dataclass(frozen=True)
class ProviderCapabilities:
    """Snapshot of a single provider's declared capabilities.

    All four fields come from ``ClassVar`` declarations on the provider
    class; keep them in sync there, not here.
    """

    native_tools: frozenset[NativeTool] = field(default_factory=frozenset)
    streaming: bool = True
    structured_output: bool = True
    thinking: bool = True

    def supports(self, tool: NativeTool) -> bool:
        return tool in self.native_tools


@lru_cache(maxsize=1)
def provider_capabilities() -> dict[str, ProviderCapabilities]:
    """Return the capability matrix for every registered provider.

    Keys are the provider names recognised by ``LLMEngine`` (the
    ``provider=`` argument and the ``LLMEngine._PROVIDER_RULES`` map);
    values are :class:`ProviderCapabilities` instances.

    Cached after first call; the underlying ``ClassVar`` declarations
    are immutable in practice so re-querying the providers each call
    would just thrash the import system.

    **Graceful degradation** — each provider class is imported lazily and
    *individually*.  If importing one provider's module fails (e.g. a
    broken optional SDK that explodes at import time), that provider is
    omitted from the returned matrix and a :class:`UserWarning` is issued,
    rather than letting one bad import break introspection for every other
    provider.
    """
    out: dict[str, ProviderCapabilities] = {}
    for name, module_path, attr in _PROVIDER_IMPORTS:
        try:
            cls: Any = getattr(importlib.import_module(module_path), attr)
        except Exception as exc:  # defend against any import-time blow-up
            warnings.warn(
                f"lazybridge.matrix: provider {name!r} is unavailable "
                f"(failed to import {module_path}.{attr}: {exc!r}); "
                f"omitting it from the capability matrix.",
                stacklevel=2,
            )
            continue
        out[name] = ProviderCapabilities(
            native_tools=frozenset(getattr(cls, "supported_native_tools", frozenset())),
            streaming=bool(getattr(cls, "supports_streaming", True)),
            structured_output=bool(getattr(cls, "supports_structured_output", True)),
            thinking=bool(getattr(cls, "supports_thinking", True)),
        )
    return out


def native_tool_support() -> dict[str, list[str]]:
    """Compact ``provider → [native-tool names]`` mapping.

    Convenient for README tables and doc generation; the full
    :class:`ProviderCapabilities` shape is what most callers want.
    """
    return {name: sorted(t.value for t in caps.native_tools) for name, caps in provider_capabilities().items()}


__all__ = [
    "ProviderCapabilities",
    "provider_capabilities",
    "native_tool_support",
]
