"""True token streaming through ``Plan.stream()`` / ``ReplanEngine.stream()``.

Pre-fix, ``Plan.stream()`` awaited the entire plan and yielded the final
text once — ``Agent(engine=plan).stream()`` silently degraded to a blocking
call.  The fix threads an *ambient token sink*
(``lazybridge/core/streaming.py``) from the composite engine's ``stream()``
down to every nested ``LLMEngine.run()``, which adopts it and streams its
turns live.

Contract under test:

* sequential LLM steps stream their tokens in step order, live (not
  buffered until the end);
* parallel bands do not stream (no token interleaving);
* nested agents-as-tools below an adopting engine stay silent;
* plans with no streaming-capable step fall back to yielding the final
  text once (the pre-fix contract);
* closing the stream early cancels the in-flight plan run.

``LLMEngine._loop`` is monkeypatched class-wide (same technique as
``test_stream_buffer_backpressure.py``): the fake pushes
``"<agent>:<i>"`` tokens into the sink it receives — which, for these
tests, is the ambient sink adopted by the real ``LLMEngine.run()``.
"""

from __future__ import annotations

import asyncio

import pytest

from lazybridge import Agent, Plan, Step
from lazybridge.engines.llm import LLMEngine
from lazybridge.envelope import Envelope

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_loop(
    monkeypatch, *, tokens: int = 2, started: list[str] | None = None, gate: dict[str, asyncio.Event] | None = None
):
    """Patch ``LLMEngine._loop`` with a fake that streams into the adopted sink.

    * appends the engine's agent name to ``started`` on entry (ordering probe);
    * pushes ``"<name>:<i>"`` for ``i in range(tokens)`` when a sink was adopted;
    * blocks on ``gate[name]`` (if present) before returning — liveness probe.
    """

    async def _fake_loop(self, env, *, _stream_sink=None, **_kw):
        name = getattr(self, "_agent_name", "agent")
        if started is not None:
            started.append(name)
        if _stream_sink is not None:
            for i in range(tokens):
                await _stream_sink.put(f"{name}:{i}")
        if gate is not None and name in gate:
            await gate[name].wait()
        return Envelope(task=env.task, payload=f"{name}-done")

    monkeypatch.setattr(LLMEngine, "_loop", _fake_loop)


def _agent(name: str) -> Agent:
    return Agent(engine=LLMEngine("claude-opus-4-7"), name=name)


async def _collect(plan: Plan) -> list[str]:
    return [
        tok async for tok in plan.stream(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
    ]


# ---------------------------------------------------------------------------
# Sequential steps stream live, in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_steps_stream_tokens_in_step_order(monkeypatch):
    _patch_loop(monkeypatch)
    plan = Plan(Step(_agent("s1")), Step(_agent("s2")))

    chunks = await _collect(plan)

    # Tokens from both steps, step order preserved — and NOT one final blob.
    assert chunks == ["s1:0", "s1:1", "s2:0", "s2:1"]


@pytest.mark.asyncio
async def test_tokens_arrive_before_plan_completes(monkeypatch):
    """Liveness: the first step's tokens reach the consumer while the
    plan is still mid-flight (step 2 not yet started) — the defining
    difference from the old buffer-then-dump behaviour.
    """
    started: list[str] = []
    gate = {"s1": asyncio.Event()}
    _patch_loop(monkeypatch, tokens=1, started=started, gate=gate)
    plan = Plan(Step(_agent("s1")), Step(_agent("s2")))

    gen = plan.stream(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
    first = await gen.__anext__()

    assert first == "s1:0"
    assert started == ["s1"]  # s2 hasn't run — we're observing mid-plan

    gate["s1"].set()
    rest = [tok async for tok in gen]
    assert rest == ["s2:0"]
    assert started == ["s1", "s2"]


# ---------------------------------------------------------------------------
# Parallel bands are silent; downstream sequential steps still stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_band_does_not_stream_but_downstream_does(monkeypatch):
    _patch_loop(monkeypatch)
    plan = Plan(
        Step(_agent("p1"), parallel=True),
        Step(_agent("p2"), parallel=True),
        Step(_agent("tail")),
    )

    chunks = await _collect(plan)

    assert chunks == ["tail:0", "tail:1"]


# ---------------------------------------------------------------------------
# Nested agents-as-tools stay silent (sink is consumed by the adopter)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_agent_below_adopting_engine_is_silent(monkeypatch):
    inner = _agent("inner")

    async def _outer_loop(self, env, *, _stream_sink=None, **_kw):
        name = getattr(self, "_agent_name", "agent")
        if name == "inner":
            # If the ambient sink leaked below the adopting frame, the
            # inner engine would receive it here and stream.
            if _stream_sink is not None:
                await _stream_sink.put("inner:LEAK")
            return Envelope(task=env.task, payload="inner-done")
        if _stream_sink is not None:
            await _stream_sink.put(f"{name}:0")
        # Simulate a tool call that invokes a nested agent mid-turn.
        await inner.run(Envelope.from_task("sub"))
        if _stream_sink is not None:
            await _stream_sink.put(f"{name}:1")
        return Envelope(task=env.task, payload=f"{name}-done")

    monkeypatch.setattr(LLMEngine, "_loop", _outer_loop)
    plan = Plan(Step(_agent("outer")))

    chunks = await _collect(plan)

    assert chunks == ["outer:0", "outer:1"]


# ---------------------------------------------------------------------------
# Fallback — no streaming-capable step yields the final text once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_function_plan_falls_back_to_final_text():
    def shout(task: str) -> str:
        return task.upper()

    plan = Plan(Step(shout))
    chunks = await _collect(plan)

    assert chunks == ["GO"]


# ---------------------------------------------------------------------------
# Early close cancels the in-flight run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_early_close_cancels_plan_run(monkeypatch):
    started: list[str] = []
    gate = {"slow": asyncio.Event()}  # never set — step 1 hangs after its token
    _patch_loop(monkeypatch, tokens=1, started=started, gate=gate)
    plan = Plan(Step(_agent("slow")), Step(_agent("after")))

    gen = plan.stream(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
    assert await gen.__anext__() == "slow:0"
    await gen.aclose()  # consumer walks away — must cancel, not hang

    await asyncio.sleep(0)  # let any (wrongly) surviving task advance
    assert started == ["slow"]  # step 2 never ran


# ---------------------------------------------------------------------------
# Construction surface
# ---------------------------------------------------------------------------


def test_plan_stream_buffer_default_and_validation():
    assert Plan().stream_buffer == 64
    assert Plan(stream_buffer=8).stream_buffer == 8
    with pytest.raises(ValueError, match="stream_buffer must be >= 1"):
        Plan(stream_buffer=0)


# ---------------------------------------------------------------------------
# Agent(engine=plan).stream() — the user-facing path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_stream_over_plan_engine_streams_tokens(monkeypatch):
    _patch_loop(monkeypatch)
    pilot = Agent(engine=Plan(Step(_agent("s1")), Step(_agent("s2"))), name="pilot")

    chunks: list[str] = []
    async for tok in pilot.stream("go"):
        chunks.append(tok)

    assert chunks == ["s1:0", "s1:1", "s2:0", "s2:1"]


# ---------------------------------------------------------------------------
# ReplanEngine.stream falls back like Plan (mock planner, no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replan_stream_yields_final_answer_without_llm():
    from lazybridge.engines.replan import PlanRound, ReplanEngine
    from lazybridge.tools import Tool

    def planner(task: str) -> PlanRound:
        return PlanRound(reasoning="trivial", tasks=[], done=True, final_answer="all done")

    planner_tool = Tool(planner, name="planner")
    eng = ReplanEngine()

    chunks = [
        tok
        async for tok in eng.stream(
            Envelope.from_task("go"), tools=[planner_tool], output_type=str, memory=None, session=None
        )
    ]
    assert chunks == ["all done"]


# ---------------------------------------------------------------------------
# Early close must not deadlock when the bounded queue is full (Codex P1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_early_close_with_full_queue_does_not_hang(monkeypatch):
    """A slow consumer that disconnects while the producer is blocked on a
    full ``stream_buffer=1`` queue must still tear down promptly: the
    runner's sentinel is skipped on cancellation, so ``aclose()`` cannot
    deadlock on ``sink.put(None)`` into a queue nobody drains."""
    _patch_loop(monkeypatch, tokens=50)  # floods the size-1 queue
    plan = Plan(Step(_agent("flood")), stream_buffer=1)

    async def _consume_one_then_close() -> None:
        gen = plan.stream(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
        assert await gen.__anext__() == "flood:0"
        await gen.aclose()

    await asyncio.wait_for(_consume_one_then_close(), timeout=5.0)


@pytest.mark.asyncio
async def test_llm_engine_stream_early_close_with_full_queue_does_not_hang(monkeypatch):
    """Same contract for ``LLMEngine.stream`` directly (the pattern the
    plan-level fix was copied from)."""
    _patch_loop(monkeypatch, tokens=50)
    eng = LLMEngine("claude-opus-4-7", stream_buffer=1)
    eng._agent_name = "flood"

    async def _consume_one_then_close() -> None:
        gen = eng.stream(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
        assert await gen.__anext__() == "flood:0"
        await gen.aclose()

    await asyncio.wait_for(_consume_one_then_close(), timeout=5.0)
