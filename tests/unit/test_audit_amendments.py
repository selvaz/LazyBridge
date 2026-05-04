"""LLMEngine runtime knobs and strict provider routing.

1. ``LLMEngine.max_turns`` default of 20 — large enough that legitimate
   tool loops don't false-positive ``MaxTurnsExceeded``.
2. ``LLMEngine.set_default_provider(None)`` opts into strict routing so
   unknown models raise at construction instead of silently falling
   back to the default provider (Anthropic by default).
3. ``BaseProvider.is_retryable(exc) -> bool | None`` provider-level
   retry classifier hook, consulted by ``Executor._should_retry``
   before the generic string/status heuristic.
"""

from __future__ import annotations

import pytest

from lazybridge.core.executor import Executor, _is_retryable
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Role,
    UsageStats,
)

# ---------------------------------------------------------------------------
# A. max_turns default bump
# ---------------------------------------------------------------------------


def test_llm_engine_default_max_turns_is_20():
    """Default tool-loop budget is 20 rounds, not 10."""
    from lazybridge.engines.llm import LLMEngine

    # Stub the anthropic SDK so LLMEngine() doesn't try to contact it.
    eng = LLMEngine("claude-opus-4-7")
    assert eng.max_turns == 20


# ---------------------------------------------------------------------------
# B. Strict-routing opt-in via set_default_provider(None)
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_provider_defaults():
    """Snapshot + restore class state so tests don't leak."""
    from lazybridge.engines.llm import LLMEngine

    original = LLMEngine._PROVIDER_DEFAULT
    yield
    LLMEngine._PROVIDER_DEFAULT = original


def test_strict_routing_raises_on_unknown_model(_restore_provider_defaults):
    """set_default_provider(None) turns unknown-model fallback into ValueError."""
    from lazybridge.engines.llm import LLMEngine

    LLMEngine.set_default_provider(None)
    with pytest.raises(ValueError, match="No provider rule matches"):
        LLMEngine._infer_provider("grok-2")


def test_strict_routing_still_matches_known_models(_restore_provider_defaults):
    """Native rules still fire under strict mode — it only affects unknowns."""
    from lazybridge.engines.llm import LLMEngine

    LLMEngine.set_default_provider(None)
    assert LLMEngine._infer_provider("claude-opus-4-7") == "anthropic"
    assert LLMEngine._infer_provider("gpt-4o") == "openai"
    assert LLMEngine._infer_provider("gemini-pro") == "google"


def test_set_default_provider_can_redirect(_restore_provider_defaults):
    """Passing a string reroutes the fallback target instead of disabling it."""
    from lazybridge.engines.llm import LLMEngine

    LLMEngine.set_default_provider("openai")
    with pytest.warns(UserWarning, match="defaulting to"):
        # "grok-2" hits no explicit rule — with openai as default, that's the answer.
        assert LLMEngine._infer_provider("grok-2") == "openai"


# ---------------------------------------------------------------------------
# C. BaseProvider.is_retryable() hook
# ---------------------------------------------------------------------------


class _ProviderStub(BaseProvider):
    """Minimal provider stub that lets each test inject a retry verdict and count calls."""

    default_model = "stub"

    def __init__(self, *, retry_verdict=None, fail_times=0, **kwargs):
        self._retry_verdict = retry_verdict
        self._fail_times = fail_times
        self._calls = 0
        super().__init__(**kwargs)

    def _init_client(self, **kwargs):
        pass

    def is_retryable(self, exc):
        return self._retry_verdict

    def complete(self, request):
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("opaque-sdk-error")
        return CompletionResponse(content="ok", usage=UsageStats())

    def stream(self, request):
        raise NotImplementedError

    async def acomplete(self, request):
        return self.complete(request)

    async def astream(self, request):
        raise NotImplementedError


def _request():
    return CompletionRequest(messages=[Message(role=Role.USER, content="x")])


def test_default_is_retryable_returns_none():
    """Default implementation is a no-op; classifier falls through to generic heuristic."""
    prov = _ProviderStub()
    assert prov.is_retryable(RuntimeError("anything")) is None


def test_provider_true_forces_retry_even_on_opaque_errors():
    """Provider returning True retries an error the generic heuristic would skip.

    ``RuntimeError("opaque-sdk-error")`` doesn't match any generic pattern
    (no 429, no 'rate limit', no 'connection' in the message).  Without the
    hook the Executor would NOT retry — surfacing the error on attempt 1.
    With is_retryable returning True, the Executor retries and succeeds.
    """
    prov = _ProviderStub(retry_verdict=True, fail_times=2)
    ex = Executor(prov, max_retries=2, retry_delay=0.0)
    # Generic heuristic on its own would not retry this:
    assert _is_retryable(RuntimeError("opaque-sdk-error")) is False
    # But with the provider's override, the Executor does retry and succeeds:
    with pytest.warns(UserWarning, match="transient error"):
        result = ex.execute(_request())
    assert result.content == "ok"
    assert prov._calls == 3


def test_provider_false_suppresses_generic_retry():
    """Provider returning False skips retry even when generic classifier would allow it."""

    class _Always429(BaseProvider):
        default_model = "stub"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._calls = 0

        def _init_client(self, **kwargs):
            pass

        def is_retryable(self, exc):
            # Opt out of retries for all exceptions this provider raises,
            # overriding the Executor's generic classifier.
            return False

        def complete(self, request):
            self._calls += 1
            exc = RuntimeError("rate limit exceeded")  # generic pattern matches
            raise exc

        def stream(self, request):
            raise NotImplementedError

        async def acomplete(self, request):
            return self.complete(request)

        async def astream(self, request):
            raise NotImplementedError

    prov = _Always429()
    ex = Executor(prov, max_retries=5, retry_delay=0.0)
    with pytest.raises(RuntimeError, match="rate limit"):
        ex.execute(_request())
    # Attempt 1 only — provider blocked the retry despite the matching string.
    assert prov._calls == 1


def test_provider_none_defers_to_generic_classifier():
    """Provider returning None leaves the decision to the generic heuristic."""

    class _Transient(BaseProvider):
        default_model = "stub"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._calls = 0

        def _init_client(self, **kwargs):
            pass

        # No override — BaseProvider.is_retryable default returns None.

        def complete(self, request):
            self._calls += 1
            if self._calls == 1:
                exc = RuntimeError("connection reset by peer")
                raise exc
            return CompletionResponse(content="ok", usage=UsageStats())

        def stream(self, request):
            raise NotImplementedError

        async def acomplete(self, request):
            return self.complete(request)

        async def astream(self, request):
            raise NotImplementedError

    prov = _Transient()
    ex = Executor(prov, max_retries=1, retry_delay=0.0)
    with pytest.warns(UserWarning, match="transient error"):
        ex.execute(_request())
    assert prov._calls == 2


@pytest.mark.asyncio
async def test_async_path_uses_same_classifier():
    """aexecute() consults _should_retry too, not just the sync path."""
    prov = _ProviderStub(retry_verdict=True, fail_times=1)
    ex = Executor(prov, max_retries=2, retry_delay=0.0)
    with pytest.warns(UserWarning, match="transient error"):
        result = await ex.aexecute(_request())
    assert result.content == "ok"
    assert prov._calls == 2
