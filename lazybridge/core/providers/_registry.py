"""Single provider-name → class registry.

The mapping used to live in two hand-maintained copies —
``Executor._resolve_provider`` (instantiating) and
``LLMEngine._provider_class`` (class-only, for capability checks) — which
had already drifted on the ``litellm`` special case.  Both now resolve
through :func:`provider_class`.

Import is lazy per provider (via ``importlib``) so resolving one
provider never pays the import cost — or the optional-dependency
requirements — of the others.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazybridge.core.providers.base import BaseProvider

#: key → (module path, class name).  Keys are the canonical provider
#: names plus their accepted aliases.
_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("lazybridge.core.providers.anthropic", "AnthropicProvider"),
    "claude": ("lazybridge.core.providers.anthropic", "AnthropicProvider"),
    "openai": ("lazybridge.core.providers.openai", "OpenAIProvider"),
    "gpt": ("lazybridge.core.providers.openai", "OpenAIProvider"),
    "google": ("lazybridge.core.providers.google", "GoogleProvider"),
    "gemini": ("lazybridge.core.providers.google", "GoogleProvider"),
    "deepseek": ("lazybridge.core.providers.deepseek", "DeepSeekProvider"),
    "lmstudio": ("lazybridge.core.providers.lmstudio", "LMStudioProvider"),
    "lm-studio": ("lazybridge.core.providers.lmstudio", "LMStudioProvider"),
    "lm_studio": ("lazybridge.core.providers.lmstudio", "LMStudioProvider"),
    "local": ("lazybridge.core.providers.lmstudio", "LMStudioProvider"),
    # LiteLLM is optional — only paid with ``pip install lazybridge[litellm]``;
    # lazy import keeps it off the hot path for every other provider.
    "litellm": ("lazybridge.core.providers.litellm", "LiteLLMProvider"),
}


def provider_class(name: str) -> type[BaseProvider] | None:
    """Resolve a provider name/alias to its class, or ``None`` if unknown.

    Does NOT instantiate — no API key needed, no SDK client created.
    """
    entry = _REGISTRY.get(name.lower().strip())
    if entry is None:
        return None
    module_path, class_name = entry
    module = importlib.import_module(module_path)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def known_provider_keys() -> list[str]:
    """Sorted list of every accepted provider name/alias."""
    return sorted(_REGISTRY)
