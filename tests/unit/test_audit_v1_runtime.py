"""Regression tests for the v1-stabilization runtime-core audit fixes.

Each test reproduces a defect found in the deep audit of the runtime core
(agent.py / engines/llm.py / core/executor.py) and certifies the fix:

* F1 — two Agents sharing one engine attributed every event to whichever
  agent was constructed last (``engine._agent_name`` stamping).  The
  identity is now bound per-invocation via a context variable.
* F2 — the streaming path rebuilt ``CompletionResponse`` without
  ``parsed``/``validation_error``, so structured output silently degraded
  to a raw string whenever the run streamed.
* F3 — ``_validate_and_retry`` returned ``ok=True`` with the raw,
  unvalidated payload after exhausting ``max_output_retries``.  It now
  returns an ``OutputValidationError`` envelope.
* F4 — ``stream()`` skipped the output guard and the fallback that
  ``run()`` enforces, and interpreted ``timeout`` per-chunk instead of
  as a total deadline.
* Executor ``_is_retryable`` retried permanent 4xx client errors whose
  message merely contained words like "timeout" or "connection".
* Memory recorded the first (rejected) draft instead of the response
  actually returned after a structured-output correction retry.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from lazybridge.agent import Agent
from lazybridge.core.executor import _is_retryable
from lazybridge.envelope import Envelope
from lazybridge.memory import Memory


class _Point(BaseModel):
    x: int
    y: int


class _StubEngine:
    """Minimal Engine double: returns queued envelopes, records call context."""

    def __init__(self, results: list[Envelope] | None = None) -> None:
        self.results = list(results or [])
        self.calls: list[dict[str, Any]] = []
        self.seen_names: list[str] = []

    async def run(self, env, *, tools, output_type, memory, session, store=None, plan_state=None):
        from lazybridge.engines.base import resolve_agent_name

        self.seen_names.append(resolve_agent_name(self, "agent"))
        self.calls.append({"env": env, "memory": memory})
        if self.results:
            return self.results.pop(0)
        return Envelope(task=env.task, payload="stub")

    async def stream(self, env, *, tools, output_type, memory, session):
        from lazybridge.engines.base import resolve_agent_name

        self.seen_names.append(resolve_agent_name(self, "agent"))
        yield "stub"


# ---------------------------------------------------------------------------
# F1 — per-invocation agent identity
# ---------------------------------------------------------------------------


def test_shared_engine_attributes_runs_to_the_running_agent():
    engine = _StubEngine(
        results=[
            Envelope(task="t", payload="a-answer"),
            Envelope(task="t", payload="b-answer"),
        ]
    )
    a = Agent(engine=engine, name="agent-a")
    b = Agent(engine=engine, name="agent-b")  # constructed last — stamps engine._agent_name

    asyncio.run(a.run("t"))
    asyncio.run(b.run("t"))

    # Before the fix both runs resolved to "agent-b" (the stamped name).
    assert engine.seen_names == ["agent-a", "agent-b"]


def test_shared_engine_stream_attributes_to_running_agent():
    engine = _StubEngine()
    a = Agent(engine=engine, name="stream-a")
    Agent(engine=engine, name="stream-b")

    async def _consume():
        return [tok async for tok in a.stream("t")]

    tokens = asyncio.run(_consume())
    assert tokens == ["stub"]
    assert engine.seen_names == ["stream-a"]


def test_direct_engine_use_still_reads_stamped_name():
    from lazybridge.engines.base import resolve_agent_name

    engine = _StubEngine()
    engine._agent_name = "manual"
    assert resolve_agent_name(engine, "agent") == "manual"


# ---------------------------------------------------------------------------
# F2 — streaming path carries structured output through
# ---------------------------------------------------------------------------


def test_stream_turn_preserves_parsed_and_validation_fields():
    from unittest.mock import MagicMock

    from lazybridge.core.types import StreamChunk, UsageStats
    from lazybridge.engines.llm import LLMEngine

    parsed_obj = _Point(x=1, y=2)
    chunks = [
        StreamChunk(delta='{"x": 1, '),
        StreamChunk(
            delta='"y": 2}',
            stop_reason="end_turn",
            is_final=True,
            usage=UsageStats(input_tokens=3, output_tokens=4),
            parsed=parsed_obj,
            validated=True,
        ),
    ]

    async def _astream(req):
        for c in chunks:
            yield c

    executor = MagicMock()
    executor.astream = _astream

    engine = LLMEngine.__new__(LLMEngine)
    engine.stream_idle_timeout = None

    async def _drive():
        sink: asyncio.Queue = asyncio.Queue()
        return await engine._stream_turn(executor, MagicMock(), sink)

    resp = asyncio.run(_drive())
    assert resp.parsed is parsed_obj
    assert resp.validated is True
    assert resp.validation_error is None
    assert resp.content == '{"x": 1, "y": 2}'


# ---------------------------------------------------------------------------
# F3 — exhausted output retries surface an error envelope
# ---------------------------------------------------------------------------


def test_output_validation_exhaustion_returns_error_envelope():
    bad = Envelope(task="t", payload="not json at all")
    engine = _StubEngine(results=[bad, bad.model_copy(), bad.model_copy()])
    agent = Agent(engine=engine, name="typed", output=_Point, max_output_retries=2)

    result = asyncio.run(agent.run("t"))

    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "OutputValidationError"
    assert result.error.retryable is False
    # The raw payload is preserved for inspection.
    assert result.payload == "not json at all"


def test_output_validator_exhaustion_returns_error_envelope():
    good_schema = Envelope(task="t", payload='{"x": 1, "y": 2}')
    engine = _StubEngine(results=[good_schema, good_schema.model_copy()])

    def _reject(p: _Point) -> _Point:
        raise ValueError("domain says no")

    agent = Agent(
        engine=engine,
        name="typed",
        output=_Point,
        output_validator=_reject,
        max_output_retries=1,
    )
    result = asyncio.run(agent.run("t"))
    assert result.ok is False
    assert result.error is not None
    assert result.error.type == "OutputValidationError"
    assert "domain says no" in result.error.message


def test_output_validation_success_still_ok():
    engine = _StubEngine(results=[Envelope(task="t", payload='{"x": 5, "y": 6}')])
    agent = Agent(engine=engine, name="typed", output=_Point)
    result = asyncio.run(agent.run("t"))
    assert result.ok is True
    assert result.payload == _Point(x=5, y=6)


# ---------------------------------------------------------------------------
# F4 — stream() pipeline parity with run()
# ---------------------------------------------------------------------------


class _GuardAction:
    def __init__(self, allowed: bool, message: str | None = None, modified_text: str | None = None):
        self.allowed = allowed
        self.message = message
        self.modified_text = modified_text


class _OutputBlockGuard:
    async def acheck_input(self, text: str) -> _GuardAction:
        return _GuardAction(True)

    async def acheck_output(self, text: str) -> _GuardAction:
        return _GuardAction("forbidden" not in text, message="output blocked")


class _StreamEngine:
    """Engine whose stream yields fixed tokens (or raises before the first)."""

    def __init__(self, tokens: list[str] | None = None, error: Exception | None = None):
        self.tokens = tokens or []
        self.error = error

    async def run(self, env, *, tools, output_type, memory, session, store=None, plan_state=None):
        return Envelope(task=env.task, payload="".join(self.tokens))

    async def stream(self, env, *, tools, output_type, memory, session):
        if self.error is not None:
            raise self.error
        for t in self.tokens:
            yield t


def test_stream_applies_output_guard_on_accumulated_text():
    agent = Agent(
        engine=_StreamEngine(tokens=["for", "bidden"]),
        name="guarded",
        guard=_OutputBlockGuard(),
    )

    async def _consume():
        return [tok async for tok in agent.stream("t")]

    with pytest.raises(ValueError, match="output blocked"):
        asyncio.run(_consume())


def test_stream_output_guard_block_skips_store_write():
    from lazybridge.store import Store

    store = Store()
    agent = Agent(
        engine=_StreamEngine(tokens=["forbidden"]),
        name="guarded",
        guard=_OutputBlockGuard(),
        store=store,
    )

    async def _consume():
        return [tok async for tok in agent.stream("t")]

    with pytest.raises(ValueError):
        asyncio.run(_consume())
    from lazybridge.sentinels import _AGENT_OUTPUT_KEY_PREFIX

    assert store.read(_AGENT_OUTPUT_KEY_PREFIX + "guarded") is None


def test_stream_falls_back_when_engine_fails_before_first_token():
    fallback = Agent(engine=_StreamEngine(tokens=["plan", " B"]), name="backup")
    agent = Agent(
        engine=_StreamEngine(error=RuntimeError("provider down")),
        name="primary",
        fallback=fallback,
    )

    async def _consume():
        return [tok async for tok in agent.stream("t")]

    tokens = asyncio.run(_consume())
    assert "".join(tokens) == "plan B"


def test_stream_does_not_fall_back_after_tokens_were_yielded():
    class _MidFailEngine(_StreamEngine):
        async def stream(self, env, *, tools, output_type, memory, session):
            yield "partial"
            raise RuntimeError("mid-stream failure")

    fallback = Agent(engine=_StreamEngine(tokens=["never"]), name="backup")
    agent = Agent(engine=_MidFailEngine(), name="primary", fallback=fallback)

    async def _consume():
        return [tok async for tok in agent.stream("t")]

    with pytest.raises(RuntimeError, match="mid-stream failure"):
        asyncio.run(_consume())


def test_stream_timeout_is_a_total_deadline():
    class _SlowEngine(_StreamEngine):
        async def stream(self, env, *, tools, output_type, memory, session):
            # Each chunk arrives well within a per-chunk window, but the
            # total run exceeds the deadline — per-chunk semantics would
            # never fire here.
            for _ in range(50):
                await asyncio.sleep(0.02)
                yield "x"

    agent = Agent(engine=_SlowEngine(), name="slow", timeout=0.15)

    async def _consume():
        return [tok async for tok in agent.stream("t")]

    with pytest.raises(TimeoutError, match="exceeded timeout"):
        asyncio.run(_consume())


def test_stream_completes_within_total_deadline():
    agent = Agent(engine=_StreamEngine(tokens=["a", "b"]), name="fast", timeout=5.0)

    async def _consume():
        return [tok async for tok in agent.stream("t")]

    assert asyncio.run(_consume()) == ["a", "b"]


# ---------------------------------------------------------------------------
# Executor retry classification
# ---------------------------------------------------------------------------


def test_client_error_with_transient_words_is_not_retryable():
    class _FakeAPIError(Exception):
        status_code = 400

    exc = _FakeAPIError("invalid 'timeout' parameter in connection settings")
    assert _is_retryable(exc) is False


def test_408_and_429_and_5xx_remain_retryable():
    class _E(Exception):
        def __init__(self, code):
            super().__init__("x")
            self.status_code = code

    assert _is_retryable(_E(408)) is True
    assert _is_retryable(_E(429)) is True
    assert _is_retryable(_E(503)) is True
    assert _is_retryable(_E(404)) is False


def test_unstructured_transient_message_still_retryable():
    exc = Exception("connection reset by peer")
    assert _is_retryable(exc) is True


# ---------------------------------------------------------------------------
# Memory ↔ structured-output retry consistency
# ---------------------------------------------------------------------------


def test_memory_amend_last_replaces_assistant_half():
    mem = Memory(strategy="none")
    mem.add("q1", "draft answer")
    mem.amend_last("final answer")
    msgs = mem.messages()
    assert msgs[-1].content == "final answer"
    assert msgs[-2].content == "q1"


def test_memory_amend_last_noop_when_empty():
    mem = Memory(strategy="none")
    mem.amend_last("nothing to amend")  # must not raise
    assert mem.messages() == []


def test_validate_retry_amends_memory_with_corrected_answer():
    class _MemoryStubEngine(_StubEngine):
        async def run(self, env, *, tools, output_type, memory, session, store=None, plan_state=None):
            result = await super().run(
                env,
                tools=tools,
                output_type=output_type,
                memory=memory,
                session=session,
            )
            # Mirror LLMEngine's contract: record the turn when a memory is
            # passed (only the first attempt gets one; retries pass None).
            if memory is not None:
                memory.add(env.task or "", result.text())
            return result

    mem = Memory(strategy="none")
    engine = _MemoryStubEngine(
        results=[
            Envelope(task="t", payload="not json"),  # first attempt — recorded
            Envelope(task="t", payload='{"x": 1, "y": 2}'),  # correction retry
        ]
    )
    agent = Agent(engine=engine, name="typed", output=_Point, memory=mem, max_output_retries=1)
    result = asyncio.run(agent.run("t"))
    assert result.ok is True
    # Memory now holds the corrected answer, not the rejected draft.
    assert mem.messages()[-1].content == '{"x": 1, "y": 2}'


# ---------------------------------------------------------------------------
# Session inheritance warning
# ---------------------------------------------------------------------------


def test_warns_when_child_pinned_to_inherited_session_of_other_orchestrator():
    from lazybridge.session import Session

    child = Agent(engine=_StubEngine(), name="shared-child")
    sess_a = Session()
    sess_b = Session()
    Agent(engine=_StubEngine(), name="orch-a", tools=[child], session=sess_a)
    with pytest.warns(UserWarning, match="already inherited a session"):
        Agent(engine=_StubEngine(), name="orch-b", tools=[child], session=sess_b)


def test_no_warning_when_child_session_set_explicitly():
    import warnings

    from lazybridge.session import Session

    sess_child = Session()
    sess_b = Session()
    child = Agent(engine=_StubEngine(), name="explicit-child", session=sess_child)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Agent(engine=_StubEngine(), name="orch", tools=[child], session=sess_b)


# ---------------------------------------------------------------------------
# Agent(verify=...) recursion — found live by the pre-v1 stress notebook
# ---------------------------------------------------------------------------


def test_agent_verify_kwarg_does_not_recurse():
    """``Agent(verify=judge).run()`` used to recurse infinitely:
    ``_run_body`` → ``verify_with_retry`` → ``agent.run()`` → ``_run_body``
    → the same verify branch again, until RecursionError.  The helper now
    runs the engine-only pipeline inside the verify loop."""
    engine = _StubEngine(results=[Envelope(task="t", payload="answer one")])

    judge_calls: list[str] = []

    def judge(text: str) -> str:
        judge_calls.append(text)
        return "approved"

    agent = Agent(engine=engine, name="verified", verify=judge)
    result = asyncio.run(agent.run("t"))
    assert result.ok and result.payload == "answer one"
    assert judge_calls == ["answer one"]
    assert len(engine.calls) == 1


def test_agent_verify_retry_feeds_judge_feedback_back():
    engine = _StubEngine(
        results=[
            Envelope(task="t", payload="draft"),
            Envelope(task="t", payload="final"),
        ]
    )
    verdicts = iter(["rejected: too short", "approved"])

    def judge(text: str) -> str:
        return next(verdicts)

    agent = Agent(engine=engine, name="verified", verify=judge, max_verify=3)
    result = asyncio.run(agent.run("t"))
    assert result.ok and result.payload == "final"
    assert len(engine.calls) == 2
    # The retry env must carry the judge's feedback as context.
    retry_env = engine.calls[1]["env"]
    assert "too short" in (retry_env.context or "")


def test_agent_verify_with_agent_judge_does_not_recurse():
    # Judge that is itself an Agent — the shape the stress notebook used.
    judge = Agent(engine=_StubEngine(results=[Envelope(task="j", payload="approved")]), name="judge")
    engine = _StubEngine(results=[Envelope(task="t", payload="content")])
    agent = Agent(engine=engine, name="verified", verify=judge)
    result = asyncio.run(agent.run("t"))
    assert result.ok and result.payload == "content"


def test_agent_verify_retry_preserves_attachments_and_payload():
    """Codex P2 on PR #110: the rebuilt retry envelope dropped the
    original env's images/audio/payload, so every post-rejection attempt
    ran without the input the first attempt had."""
    from lazybridge.core.types import ImageContent

    engine = _StubEngine(
        results=[
            Envelope(task="t", payload="draft"),
            Envelope(task="t", payload="final"),
        ]
    )
    verdicts = iter(["rejected: look at the image again", "approved"])
    agent = Agent(engine=engine, name="verified", verify=lambda text: next(verdicts), max_verify=3)

    img = ImageContent(base64_data="aGk=", media_type="image/png")
    original = Envelope(task="describe", images=[img], payload="user-input")
    result = asyncio.run(agent.run(original))

    assert result.ok and result.payload == "final"
    assert len(engine.calls) == 2
    retry_env = engine.calls[1]["env"]
    assert retry_env.images == [img], "images dropped on verify retry"
    assert retry_env.payload == "user-input", "payload dropped on verify retry"
    assert "look at the image again" in (retry_env.context or "")
