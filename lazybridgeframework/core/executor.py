"""Executor — clean execution layer over any LLM provider.

Handles:
  - Provider resolution (string name → concrete BaseProvider)
  - Retry with exponential backoff on transient errors
  - Sync and async execution + streaming

No memory, no tracking, no context injection — those live in LazyAgent/LazySession.
"""

from __future__ import annotations

import asyncio
import random
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from lazybridgeframework.core.providers.base import BaseProvider
from lazybridgeframework.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)

# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------

def _resolve_provider(
    provider: str | BaseProvider,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    if isinstance(provider, BaseProvider):
        return provider

    from lazybridgeframework.core.providers.anthropic import AnthropicProvider
    from lazybridgeframework.core.providers.deepseek import DeepSeekProvider
    from lazybridgeframework.core.providers.google import GoogleProvider
    from lazybridgeframework.core.providers.openai import OpenAIProvider

    registry: dict[str, type[BaseProvider]] = {
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "deepseek": DeepSeekProvider,
    }
    key = provider.lower().strip()
    if key not in registry:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: {', '.join(sorted(set(registry.keys())))}."
        )
    return registry[key](api_key=api_key, model=model, **kwargs)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    for attr in ("status_code", "status", "http_status", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and (code == 429 or 500 <= code < 600):
            return True
    s = str(exc).lower()
    patterns = ("rate limit", "ratelimit", "too many requests",
                "server error", "service unavailable", "timeout",
                "connection", "network", "502", "503", "504")
    return any(p in s for p in patterns)


# Warning template reused by both sync and async retry loops
_RETRY_WARN = (
    "Executor: transient error on attempt {attempt}/{total} "
    "({exc_type}: {exc}). Retrying in {delay:.1f}s."
)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Thin, stateless execution layer over a provider.

    LazyAgent builds on top of this — don't use directly unless you need
    raw provider access.
    """

    def __init__(
        self,
        provider: str | BaseProvider,
        *,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self._provider = _resolve_provider(provider, api_key, model, **kwargs)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    @property
    def model(self) -> str:
        return self._provider.model

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def execute(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a completion request, retrying on transient errors."""
        for attempt in range(self._max_retries + 1):
            try:
                return self._provider.complete(request)
            except Exception as exc:
                if attempt >= self._max_retries or not _is_retryable(exc):
                    raise
                # exponential backoff: base_delay * 2^attempt, with ±10% random jitter
                delay = self._retry_delay * (2 ** attempt) * (0.9 + random.random() * 0.2)
                warnings.warn(
                    _RETRY_WARN.format(
                        attempt=attempt + 1, total=self._max_retries + 1,
                        exc_type=type(exc).__name__, exc=exc, delay=delay,
                    ),
                    UserWarning, stacklevel=2,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Stream a completion request (no retry — streaming is not idempotent)."""
        return self._provider.stream(request)

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def aexecute(self, request: CompletionRequest) -> CompletionResponse:
        """Async version of execute(), retrying on transient errors."""
        for attempt in range(self._max_retries + 1):
            try:
                return await self._provider.acomplete(request)
            except Exception as exc:
                if attempt >= self._max_retries or not _is_retryable(exc):
                    raise
                # exponential backoff: base_delay * 2^attempt, with ±10% random jitter
                delay = self._retry_delay * (2 ** attempt) * (0.9 + random.random() * 0.2)
                warnings.warn(
                    _RETRY_WARN.format(
                        attempt=attempt + 1, total=self._max_retries + 1,
                        exc_type=type(exc).__name__, exc=exc, delay=delay,
                    ),
                    UserWarning, stacklevel=2,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming version (no retry — streaming is not idempotent)."""
        async for chunk in self._provider.astream(request):
            yield chunk
