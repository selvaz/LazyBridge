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

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from lazybridge.core.types import NativeTool


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
    """
    # Import lazily so this module stays free of optional-dep cost.
    from lazybridge.core.providers.anthropic import AnthropicProvider
    from lazybridge.core.providers.deepseek import DeepSeekProvider
    from lazybridge.core.providers.google import GoogleProvider
    from lazybridge.core.providers.litellm import LiteLLMProvider
    from lazybridge.core.providers.lmstudio import LMStudioProvider
    from lazybridge.core.providers.openai import OpenAIProvider

    classes: list[tuple[str, Any]] = [
        ("anthropic", AnthropicProvider),
        ("openai", OpenAIProvider),
        ("google", GoogleProvider),
        ("deepseek", DeepSeekProvider),
        ("litellm", LiteLLMProvider),
        ("lmstudio", LMStudioProvider),
    ]
    out: dict[str, ProviderCapabilities] = {}
    for name, cls in classes:
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
