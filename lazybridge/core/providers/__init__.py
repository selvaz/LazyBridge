"""Provider classes — import from here for convenience.

The concrete provider classes are loaded lazily (PEP 562): importing this package
(and therefore ``import lazybridge``) does not import any provider SDK. The SDK of
a provider is imported only when its class is first accessed, for example
``from lazybridge.core.providers import AnthropicProvider``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from lazybridge.core.providers.base import (
    BaseProvider,
    Tier,
    UnsupportedFeatureError,
    UnsupportedNativeToolError,
)

if TYPE_CHECKING:
    from lazybridge.core.providers.anthropic import AnthropicProvider
    from lazybridge.core.providers.deepseek import DeepSeekProvider
    from lazybridge.core.providers.google import GoogleProvider
    from lazybridge.core.providers.lmstudio import LMStudioProvider
    from lazybridge.core.providers.openai import OpenAIProvider

_LAZY_PROVIDERS = {
    "AnthropicProvider": "lazybridge.core.providers.anthropic",
    "DeepSeekProvider": "lazybridge.core.providers.deepseek",
    "GoogleProvider": "lazybridge.core.providers.google",
    "LMStudioProvider": "lazybridge.core.providers.lmstudio",
    "OpenAIProvider": "lazybridge.core.providers.openai",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_PROVIDERS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_PROVIDERS))


__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "DeepSeekProvider",
    "GoogleProvider",
    "LMStudioProvider",
    "OpenAIProvider",
    "Tier",
    "UnsupportedFeatureError",
    "UnsupportedNativeToolError",
]
