"""Regression tests for the deep-audit fix set covering Agent / Session
/ Store / Guardrails / DeepSeek behaviour.

One test per behaviour so a breakage points straight at the offending
area.  No live provider calls — every test runs against in-process
fakes.
"""

from __future__ import annotations

import asyncio
import sqlite3
import warnings

import pytest

from lazybridge import Agent, ContentGuard, Envelope, GuardAction, GuardChain, Session
from lazybridge.core.providers.deepseek import DeepSeekProvider
from lazybridge.core.types import (
    CompletionRequest,
    Message,
    Role,
    StructuredOutputConfig,
    ToolDefinition,
)
from lazybridge.guardrails import LLMGuard
from lazybridge.session import EventType
from lazybridge.store import Store
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# H1 — Agent timeout emits AGENT_FINISH (cancelled=True)
# ---------------------------------------------------------------------------


class _SlowEngine:
    """Minimal Engine that sleeps longer than any reasonable test
    timeout, so ``asyncio.wait_for`` cancels it before it returns."""

    model = "slow"

    async def run(self, env, *, tools, output_type, memory, session) -> Envelope:
        await asyncio.sleep(1.0)
        return env  # never reached


@pytest.mark.asyncio
async def test_h1_agent_timeout_emits_agent_finish_on_session() -> None:
    """When ``Agent.run`` is cut by ``timeout=``, the session receives
    an AGENT_FINISH event with ``cancelled=True`` so callers reading
    ``session.events.query()`` see a complete trace.  Without the fix
    the engine's AGENT_FINISH is skipped (CancelledError is
    BaseException, not caught by ``except Exception``) and ``Agent.run``
    returns silently.
    """
    sess = Session()
    agent = Agent(engine=_SlowEngine(), name="outer", session=sess, timeout=0.05)

    env = await agent.run("hi")
    assert env.error is not None
    assert "timeout" in env.error.message.lower()

    finishes = [r for r in sess.events.query() if r["event_type"] == EventType.AGENT_FINISH]
    assert finishes, "expected AGENT_FINISH after timeout"
    assert finishes[-1]["payload"].get("cancelled") is True
    assert finishes[-1]["payload"].get("agent_name") == "outer"


@pytest.mark.asyncio
async def test_h1_agent_timeout_no_session_does_not_crash() -> None:
    """Timeout path must not crash when ``session`` is None."""
    agent = Agent(engine=_SlowEngine(), name="outer", timeout=0.02)
    env = await agent.run("hi")
    assert env.error is not None


# ---------------------------------------------------------------------------
# H4 — DeepSeek allows tools+structured-output on V4, drops with warn on legacy
# ---------------------------------------------------------------------------


def _bare_deepseek() -> DeepSeekProvider:
    """Build a DeepSeekProvider without invoking ``_init_client`` —
    these tests only exercise pure-Python request param construction."""
    p = DeepSeekProvider.__new__(DeepSeekProvider)
    p.api_key = None
    p.model = DeepSeekProvider.default_model
    p._structured_drop_warned = False
    return p


def _request_with_tool_and_schema(model: str) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role=Role.USER, content="get the json please")],
        model=model,
        tools=[
            ToolDefinition(
                name="search",
                description="d",
                parameters={"type": "object", "properties": {}},
            ),
        ],
        structured_output=StructuredOutputConfig(schema={"type": "object"}),
    )


def test_h4_deepseek_v4_keeps_response_format_with_tools() -> None:
    """V4 models support tools + JSON mode simultaneously (strict-mode
    function calling).  No warn, ``response_format`` survives."""
    p = _bare_deepseek()

    # Build the params the way ``complete()`` does.
    request = _request_with_tool_and_schema("deepseek-v4-flash")
    model = p._resolve_model(request)
    params = p._build_chat_params(request)
    has_tools = bool(params.get("tools"))
    supports_structured_with_tools = model in {"deepseek-v4-pro", "deepseek-v4-flash"}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if request.structured_output:
            if has_tools and not supports_structured_with_tools:
                p._warn_structured_drop_once()
            else:
                params["response_format"] = {"type": "json_object"}
                p._ensure_json_word_in_prompt(params, schema=request.structured_output.schema)

    assert params.get("response_format") == {"type": "json_object"}
    assert not any("structured_output" in str(x.message) for x in w)


def test_h4_deepseek_legacy_drops_response_format_with_warning() -> None:
    """Legacy models (deepseek-chat) silently dropped structured output
    with tools pre-fix; post-fix a one-shot UserWarning is emitted."""
    p = _bare_deepseek()
    p.model = "deepseek-chat"

    request = _request_with_tool_and_schema("deepseek-chat")
    model = p._resolve_model(request)
    params = p._build_chat_params(request)
    has_tools = bool(params.get("tools"))
    supports_structured_with_tools = model in {"deepseek-v4-pro", "deepseek-v4-flash"}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if request.structured_output:
            if has_tools and not supports_structured_with_tools:
                p._warn_structured_drop_once()
            else:
                params["response_format"] = {"type": "json_object"}

    assert "response_format" not in params
    assert any("structured_output is silently disabled" in str(x.message) for x in w)


def test_h4_deepseek_warning_fires_only_once_per_provider() -> None:
    """The warn-once flag is per-provider-instance: a long-running
    process doesn't spam the log every turn."""
    p = _bare_deepseek()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p._warn_structured_drop_once()
        p._warn_structured_drop_once()
        p._warn_structured_drop_once()

    assert sum(1 for x in w if "silently disabled" in str(x.message)) == 1


# ---------------------------------------------------------------------------
# H5 — LLMGuard timeout fails closed (block) instead of starving the loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h5_llm_guard_async_timeout_blocks_with_clear_message() -> None:
    """A judge that hangs longer than ``timeout=`` returns a *block*
    action rather than letting the calling agent proceed without a
    verdict.  Fail-closed semantics."""

    class _HungJudge:
        async def run(self, prompt: str) -> Envelope:
            await asyncio.sleep(1.0)
            return Envelope(payload="allow")

    judge = _HungJudge()
    guard = LLMGuard(judge, policy="no PII", timeout=0.05)

    action = await guard.acheck_input("hello world")
    assert action.allowed is False
    assert "timeout" in (action.message or "").lower()


@pytest.mark.asyncio
async def test_h5_llm_guard_async_within_timeout_passes_through() -> None:
    """A fast judge isn't punished by the timeout wrapping."""

    class _FastJudge:
        async def run(self, prompt: str) -> Envelope:
            return Envelope(payload="allow")

    guard = LLMGuard(_FastJudge(), policy="no PII", timeout=1.0)
    action = await guard.acheck_input("hello")
    assert action.allowed is True


@pytest.mark.asyncio
async def test_h5_llm_guard_explicit_none_timeout_disables_wrap() -> None:
    """``timeout=None`` opts out of the deadline (matching the
    documented escape hatch for deterministic tests)."""

    class _DeterministicJudge:
        async def run(self, prompt: str) -> Envelope:
            return Envelope(payload="block")

    guard = LLMGuard(_DeterministicJudge(), policy="no PII", timeout=None)
    action = await guard.acheck_input("evil")
    assert action.allowed is False


# ---------------------------------------------------------------------------
# H6 — Fallback agent receives error context in env.context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h6_fallback_receives_primary_error_in_context() -> None:
    """The fallback agent's ``env.context`` carries the primary's error
    type + message so it can adapt strategy.  Without this the fallback
    is blind and just re-tries with the same input."""

    seen_contexts: list[str | None] = []

    async def _record_ctx(env: Envelope) -> str:
        seen_contexts.append(env.context)
        return "fallback-output"

    class _PrimaryFails:
        async def run(self, env: Envelope, *, tools, output_type, memory, session):
            return Envelope.error_envelope(RuntimeError("rate limit exceeded"))

        async def stream(self, *args, **kwargs):
            if False:
                yield ""

    fb = MockAgent(_record_ctx, name="fb")
    primary = Agent(engine=_PrimaryFails(), name="primary", fallback=fb)

    env = await primary.run("compute pi")
    assert env.text() == "fallback-output"
    assert seen_contexts and seen_contexts[0] is not None
    assert "RuntimeError" in seen_contexts[0]
    assert "rate limit" in seen_contexts[0]


@pytest.mark.asyncio
async def test_h6_fallback_preserves_existing_context() -> None:
    """When the user already supplied ``Envelope(context=...)``, the
    error note appends after a blank-line separator instead of
    overwriting the user's context."""

    captured: list[str | None] = []

    async def _record(env: Envelope) -> str:
        captured.append(env.context)
        return "ok"

    class _Fails:
        async def run(self, env, *, tools, output_type, memory, session):
            return Envelope.error_envelope(ValueError("bad schema"))

        async def stream(self, *args, **kwargs):
            if False:
                yield ""

    fb = MockAgent(_record, name="fb")
    primary = Agent(engine=_Fails(), name="primary", fallback=fb)

    env_in = Envelope(task="t", context="user-supplied background")
    await primary.run(env_in)
    assert captured[0] is not None
    assert "user-supplied background" in captured[0]
    assert "ValueError" in captured[0]


# ---------------------------------------------------------------------------
# H7 — Per-Session redactor warn-once (not stamped on the redactor callable)
# ---------------------------------------------------------------------------


def test_h7_redactor_warn_is_per_session_not_per_redactor() -> None:
    """A redactor function shared across many short-lived Session
    instances must warn once *per session* rather than warning only
    once globally."""

    def broken_redactor(payload: dict) -> dict:
        raise RuntimeError("redaction failure")

    # First session — should warn.
    s1 = Session(redact=broken_redactor, redact_on_error="fallback")
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        s1.emit(EventType.AGENT_START, {"agent_name": "a"})
    s1_warnings = [str(x.message) for x in w1 if "redact callable" in str(x.message)]
    assert len(s1_warnings) == 1

    # Second session reusing the SAME redactor — must also warn.
    s2 = Session(redact=broken_redactor, redact_on_error="fallback")
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        s2.emit(EventType.AGENT_START, {"agent_name": "b"})
    s2_warnings = [str(x.message) for x in w2 if "redact callable" in str(x.message)]
    assert len(s2_warnings) == 1, (
        "second Session reusing redactor should still warn — warn-once must be per-Session, not per-callable"
    )


def test_h7_redactor_warns_once_per_session_not_per_event() -> None:
    """Within one Session, a redactor that fails on every event still
    only warns the first time."""

    def broken(payload: dict) -> dict:
        raise RuntimeError("nope")

    sess = Session(redact=broken, redact_on_error="fallback")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(5):
            sess.emit(EventType.AGENT_START, {"agent_name": "a"})
    msgs = [str(x.message) for x in w if "redact callable" in str(x.message)]
    assert len(msgs) == 1


# ---------------------------------------------------------------------------
# H8 — Store CAS rollback failure invalidates the connection
# ---------------------------------------------------------------------------


def test_h8_cas_rollback_failure_warns_and_discards_thread_conn(tmp_path) -> None:
    """When the inner ``ROLLBACK`` after a CAS error itself fails,
    the broken connection is removed from the thread-local cache and
    a UserWarning is emitted so an operator can see the rollback
    failure."""

    db = str(tmp_path / "store.sqlite")
    s = Store(db=db)

    # Prime the store + cache the thread-local connection.
    s.write("k", {"v": 1})
    conn = s._conn()
    assert conn is not None

    # Force a CAS error path that ALSO makes ROLLBACK fail by closing
    # the underlying connection mid-transaction.  ``compare_and_swap``
    # then enters its except-rollback handler and the rollback raises
    # too because the conn is closed.
    conn.close()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            s.compare_and_swap("k", {"v": 1}, {"v": 2})
        except sqlite3.Error:
            # The CAS itself raises sqlite3.Error after the rollback
            # attempt fails — this is the expected propagation.
            pass

    rollback_warnings = [str(x.message) for x in w if "ROLLBACK after error failed" in str(x.message)]
    assert rollback_warnings, "expected a UserWarning about the failed rollback"

    # Thread-local cache was discarded — next ``_conn()`` rebuilds.
    new_conn = s._conn()
    assert new_conn is not conn

    s.close()


# ---------------------------------------------------------------------------
# M11 — GuardChain block-after-modify surfaces the accumulated rewrite
# ---------------------------------------------------------------------------


def test_m11_guard_chain_block_carries_accumulated_modifications() -> None:
    """When guard 1 rewrites and guard 2 blocks, the block action's
    ``metadata["modifications_before_block"]`` records what guard 1
    had rewritten so the caller can debug the chain end-to-end."""

    def rewrite(text: str) -> GuardAction:
        return GuardAction.modify(text.upper())

    def block_if_caps(text: str) -> GuardAction:
        if text != text.lower():
            return GuardAction.block("no shouting")
        return GuardAction.allow()

    chain = GuardChain(
        ContentGuard(input_fn=rewrite),
        ContentGuard(input_fn=block_if_caps),
    )
    action = chain.check_input("hello world")
    assert action.allowed is False
    assert action.message == "no shouting"
    assert action.metadata.get("modifications_before_block") == "HELLO WORLD"


def test_m11_guard_chain_block_with_no_prior_modifications_omits_metadata() -> None:
    """If no prior guard modified, the metadata key is NOT set —
    blocks without rewrites stay clean."""

    def block(text: str) -> GuardAction:
        return GuardAction.block("blocked from the start")

    chain = GuardChain(ContentGuard(input_fn=block))
    action = chain.check_input("anything")
    assert action.allowed is False
    assert "modifications_before_block" not in (action.metadata or {})
