"""Tests covering audit-derived hardening of Plan + LLMEngine.stream.

* PlanCompiler rejects duplicate step names.
* PlanCompiler rejects ``from_step()`` references to future / same-position steps.
* Plan.run returns a ``MaxIterationsExceeded`` error envelope when the
  routing loop exhausts ``max_iterations``, instead of a success-shaped
  partial result.
* LLMEngine.stream emits AGENT_FINISH (with ``cancelled``) and cancels
  the background loop when the consumer breaks early.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lazybridge.core.executor import Executor
from lazybridge.core.providers.base import BaseProvider
from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    UsageStats,
)
from lazybridge.engines.llm import LLMEngine
from lazybridge.engines.plan import Plan, PlanCompileError, PlanCompiler, Step
from lazybridge.envelope import Envelope
from lazybridge.sentinels import from_step
from lazybridge.session import EventType, Session

# ---------------------------------------------------------------------------
# PlanCompiler hardening
# ---------------------------------------------------------------------------


def _noop_tool_step(name: str, **kw: Any) -> Step:
    return Step(target=lambda: name, name=name, **kw)


def _validate(steps: list[Step]) -> None:
    """Invoke PlanCompiler the same way Agent.__init__ does."""
    PlanCompiler().validate(steps, tool_map={})


def test_plan_compiler_rejects_duplicate_step_names() -> None:
    with pytest.raises(PlanCompileError, match="duplicate step name"):
        _validate(
            [
                _noop_tool_step("a"),
                _noop_tool_step("b"),
                _noop_tool_step("a"),  # duplicate
            ]
        )


def test_plan_compiler_rejects_forward_from_step_reference() -> None:
    """from_step() must point at a step that comes before the using step."""
    with pytest.raises(PlanCompileError, match=r"from_step.*not earlier"):
        _validate(
            [
                _noop_tool_step("a"),
                _noop_tool_step("b", task=from_step("c")),
                _noop_tool_step("c"),
            ]
        )


def test_plan_compiler_rejects_self_from_step_reference() -> None:
    """A step referencing itself via from_step() is also forward-pointing."""
    with pytest.raises(PlanCompileError, match=r"from_step.*not earlier"):
        _validate(
            [
                _noop_tool_step("a"),
                _noop_tool_step("b", task=from_step("b")),
            ]
        )


def test_plan_compiler_accepts_backward_from_step_reference() -> None:
    """Backward ``from_step()`` is the supported case — must compile."""
    _validate(
        [
            _noop_tool_step("a"),
            _noop_tool_step("b", task=from_step("a")),
        ]
    )


# ---------------------------------------------------------------------------
# Plan.run — max_iterations exhaustion produces an error envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_max_iterations_returns_error_envelope() -> None:
    """A plan whose loop body keeps routing back exhausts max_iterations
    and surfaces a ``MaxIterationsExceeded`` error envelope, NOT a
    success-shaped partial result.

    Drives the routing cycle by directly invoking the engine loop with
    a step_map whose ``_routing`` always points back at the current
    step — that cleanly exercises the iteration cap without depending
    on any LLM / tool runtime details.
    """
    plan = Plan.__new__(Plan)
    plan.steps = []
    plan.max_iterations = 3
    plan.store = None
    plan.checkpoint_key = None
    plan.resume = False
    plan.on_concurrent = "fail"

    # Patch _routing so step "a" always routes back to itself —
    # guaranteed iteration-cap exhaustion.
    plan._routing = lambda result_env, step, step_map, **kw: ("a", True)  # type: ignore[assignment]

    # Patch _exec_step to return a simple envelope unchanged.
    async def _fake_exec(step, step_env, **kw):  # type: ignore[no-untyped-def]
        return step_env

    plan._exec_step = _fake_exec  # type: ignore[assignment]
    plan._save_checkpoint = lambda **kw: None  # type: ignore[assignment]
    plan._aggregate_nested_metadata = lambda env, history: env  # type: ignore[assignment]

    # Build a step manually and seed step_map.
    step_a = Step(target=lambda: "noop", name="a")
    plan.steps = [step_a]

    env = Envelope.from_task("anything")
    result = await plan.run(env, tools=[], output_type=str, memory=None, session=None)
    assert result.error is not None
    assert result.error.type == "MaxIterationsExceeded"
    assert "max_iterations=3" in result.error.message


# ---------------------------------------------------------------------------
# LLMEngine.stream — AGENT_FINISH + cancellation
# ---------------------------------------------------------------------------


class _SlowStreamProvider(BaseProvider):
    """Streams 50 short tokens with a small sleep — gives consumers time to break."""

    default_model = "fake-stream"

    def _init_client(self, **kwargs: Any) -> None:
        pass

    def complete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        return CompletionResponse(content="", usage=UsageStats(), model=self.model)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        return self.complete(request)

    def stream(self, request: CompletionRequest):  # pragma: no cover - sync unused
        yield StreamChunk(delta="", stop_reason="stop", is_final=True)

    async def astream(self, request: CompletionRequest):
        for i in range(50):
            await asyncio.sleep(0.005)
            yield StreamChunk(delta=f"{i} ")
        yield StreamChunk(stop_reason="end_turn", is_final=True, usage=UsageStats())


def _build_stream_engine(provider: BaseProvider) -> LLMEngine:
    engine = LLMEngine.__new__(LLMEngine)
    engine._agent_name = "stream-agent"
    engine.model = provider.default_model
    engine.provider = "anthropic"
    engine.system = None
    engine.max_turns = 5
    engine.tool_choice = "auto"
    engine.temperature = None
    engine.thinking = False
    engine.request_timeout = None
    engine.max_retries = 0
    engine.retry_delay = 0.0
    engine.native_tools = []
    fake_exec = Executor(provider, max_retries=0)
    engine._make_executor = lambda: fake_exec  # type: ignore[method-assign]
    return engine


@pytest.mark.asyncio
async def test_stream_emits_agent_finish_on_normal_completion() -> None:
    sess = Session()
    engine = _build_stream_engine(_SlowStreamProvider(model="fake-stream"))
    env = Envelope.from_task("hi")
    consumed: list[str] = []
    async for tok in engine.stream(env, tools=[], output_type=str, memory=None, session=sess):
        consumed.append(tok)
    finishes = [r for r in sess.events.query() if r["event_type"] == EventType.AGENT_FINISH]
    assert finishes, "expected AGENT_FINISH event on normal stream completion"
    assert finishes[-1]["payload"].get("cancelled") is False
    sess.close()


@pytest.mark.asyncio
async def test_stream_cancels_background_loop_on_consumer_break() -> None:
    """Breaking out of the stream early cancels the loop instead of awaiting it.

    Async-generator cleanup runs on ``aclose()``, which CPython only
    schedules at GC time when the generator goes out of scope —
    explicitly close it so the test observes the cleanup deterministically.
    """
    sess = Session()
    engine = _build_stream_engine(_SlowStreamProvider(model="fake-stream"))
    env = Envelope.from_task("hi")
    tokens: list[str] = []
    agen = engine.stream(env, tools=[], output_type=str, memory=None, session=sess)
    try:
        async for tok in agen:
            tokens.append(tok)
            if len(tokens) >= 3:
                break  # early break — should cancel the loop
    finally:
        await agen.aclose()
    finishes = [r for r in sess.events.query() if r["event_type"] == EventType.AGENT_FINISH]
    assert finishes, "expected AGENT_FINISH event even on early break"
    assert finishes[-1]["payload"].get("cancelled") is True
    sess.close()
