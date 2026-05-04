"""Executor — clean execution layer over any LLM provider.

Handles:
  - Provider resolution (string name → concrete BaseProvider)
  - Retry with exponential backoff on transient errors
  - Sync and async execution + streaming

No memory, no tracking, no context injection — those live in Agent/Session.
"""

from __future__ import annotations

import asyncio
import random
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any

from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
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

    from lazybridge.core.providers.anthropic import AnthropicProvider
    from lazybridge.core.providers.deepseek import DeepSeekProvider
    from lazybridge.core.providers.google import GoogleProvider
    from lazybridge.core.providers.lmstudio import LMStudioProvider
    from lazybridge.core.providers.openai import OpenAIProvider

    registry: dict[str, type[BaseProvider]] = {
        "anthropic": AnthropicProvider,
        "claude": AnthropicProvider,
        "openai": OpenAIProvider,
        "gpt": OpenAIProvider,
        "google": GoogleProvider,
        "gemini": GoogleProvider,
        "deepseek": DeepSeekProvider,
        "lmstudio": LMStudioProvider,
        "lm-studio": LMStudioProvider,
        "lm_studio": LMStudioProvider,
        "local": LMStudioProvider,
    }
    key = provider.lower().strip()

    # LiteLLM is optional; import lazily so the provider module isn't a
    # hard dependency for every Agent() construction. Only paid with
    # ``pip install lazybridge[litellm]``.
    if key == "litellm":
        from lazybridge.core.providers.litellm import LiteLLMProvider

        return LiteLLMProvider(api_key=api_key, model=model, **kwargs)

    if key not in registry:
        raise ValueError(f"Unknown provider '{provider}'. Supported: {', '.join(sorted(set(registry.keys())))}.")
    return registry[key](api_key=api_key, model=model, **kwargs)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


# Provider SDKs converge on a small set of class names for transient
# errors.  Matching by ``__class__.__name__`` avoids importing every
# SDK (no runtime dep added) and survives string-message drift /
# non-EN locales.  Cross-checked against openai, anthropic,
# google.genai, deepseek (openai-clone), litellm.
_RETRYABLE_EXC_CLASSES = frozenset(
    {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "APIStatusError",  # SDK-generic 5xx wrapper
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "GatewayTimeoutError",
        "TimeoutError",  # builtins.TimeoutError, asyncio.TimeoutError
        "ConnectionError",  # builtins.ConnectionError + subclasses
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ConnectionRefusedError",
        "ReadTimeout",  # httpx
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "ProtocolError",
    }
)


def _is_retryable(exc: Exception) -> bool:
    # 1. Structured: HTTP-status-bearing exceptions.  Cheapest and most
    #    reliable signal across SDKs.
    for attr in ("status_code", "status", "http_status", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int) and (code == 429 or 500 <= code < 600):
            return True
    # 2. Structured: known transient class names (provider-SDK agnostic).
    #    Walk the MRO so subclasses (``ConnectionResetError`` →
    #    ``ConnectionError``) match without per-class enumeration.
    for cls in type(exc).__mro__:
        if cls.__name__ in _RETRYABLE_EXC_CLASSES:
            return True
    # 3. Last-resort string scan.  Kept for SDKs that wrap upstream
    #    errors as plain ``Exception`` / ``RuntimeError`` with the
    #    transient signal only in the message.
    s = str(exc).lower()
    patterns = (
        "rate limit",
        "ratelimit",
        "too many requests",
        "server error",
        "service unavailable",
        "timeout",
        "connection",
        "network",
        "502",
        "503",
        "504",
    )
    return any(p in s for p in patterns)


# Warning template reused by both sync and async retry loops
_RETRY_WARN = "Executor: transient error on attempt {attempt}/{total} ({exc_type}: {exc!r}). Retrying in {delay:.1f}s."


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """Thin, stateless execution layer over a provider.

    Agent / LLMEngine build on top of this — don't use directly unless you need
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

    def _should_retry(self, exc: BaseException) -> bool:
        """Consult the provider's ``is_retryable`` hook, then fall back.

        The provider's classifier wins when it returns a bool; ``None`` means
        "no opinion" and hands control to the generic heuristic.  This lets
        provider adapters (e.g. a Bedrock subclass that inspects structured
        throttling codes) express retry policy without having to patch the
        Executor.
        """
        verdict = self._provider.is_retryable(exc)
        if verdict is True:
            return True
        if verdict is False:
            return False
        # ``_is_retryable`` only inspects ``Exception`` subclasses; non-Exception
        # ``BaseException``s (KeyboardInterrupt, SystemExit) shouldn't be
        # retried by definition.
        if not isinstance(exc, Exception):
            return False
        return _is_retryable(exc)

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def execute(self, request: CompletionRequest) -> CompletionResponse:
        """Execute a completion request, retrying on transient errors."""
        for attempt in range(self._max_retries + 1):
            try:
                return self._provider.complete(request)
            except Exception as exc:
                if attempt >= self._max_retries or not self._should_retry(exc):
                    raise
                # exponential backoff: base_delay * 2^attempt, with ±10% random jitter
                delay = self._retry_delay * (2**attempt) * (0.9 + random.random() * 0.2)
                warnings.warn(
                    _RETRY_WARN.format(
                        attempt=attempt + 1,
                        total=self._max_retries + 1,
                        exc_type=type(exc).__name__,
                        exc=exc,
                        delay=delay,
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

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
                if attempt >= self._max_retries or not self._should_retry(exc):
                    raise
                # exponential backoff: base_delay * 2^attempt, with ±10% random jitter
                delay = self._retry_delay * (2**attempt) * (0.9 + random.random() * 0.2)
                warnings.warn(
                    _RETRY_WARN.format(
                        attempt=attempt + 1,
                        total=self._max_retries + 1,
                        exc_type=type(exc).__name__,
                        exc=exc,
                        delay=delay,
                    ),
                    UserWarning,
                    stacklevel=2,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Async streaming version (no retry — streaming is not idempotent)."""
        async for chunk in self._provider.astream(request):
            yield chunk
