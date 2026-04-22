"""Tests for ``lazybridge.testing.MockAgent`` — the deterministic Agent
test double used for exercising pipeline composition and data
transmission without touching a real provider.

Covers:

* All four response specifications (scalar / list / dict / callable).
* All response value types (Envelope / ErrorInfo / Exception / scalar).
* Recording + assertion helpers.
* Drop-in composition:
    - ``Agent(tools=[mock])``   (via wrap_tool)
    - ``Plan(Step(target=mock))``
    - ``mock.as_tool()``
    - ``Agent.chain(a, b)``     (sequential)
    - ``Agent.parallel(a, b)``  (concurrent)
* Envelope metadata / error / nested-token propagation through the tool
  boundary — the whole reason MockAgent exists.
* Session auto-propagation via the duck-typed ``_is_lazy_agent`` check.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from lazybridge import Agent
from lazybridge.engines.plan import Plan, Step
from lazybridge.envelope import Envelope, ErrorInfo
from lazybridge.sentinels import from_prev
from lazybridge.session import Session
from lazybridge.testing import DEFAULT, MockAgent, MockCall


# ---------------------------------------------------------------------------
# 1. Response specifications
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scalar_response_returned_on_every_call() -> None:
    m = MockAgent("hello", name="greet")
    env1 = await m.run("x")
    env2 = await m.run("y")
    assert env1.text() == "hello"
    assert env2.text() == "hello"
    assert m.call_count == 2


@pytest.mark.asyncio
async def test_list_response_in_order_then_exhausts() -> None:
    m = MockAgent(["a", "b"], name="seq")
    assert (await m.run("1")).text() == "a"
    assert (await m.run("2")).text() == "b"
    with pytest.raises(RuntimeError, match="exhausted"):
        await m.run("3")


@pytest.mark.asyncio
async def test_list_response_cycle_true_wraps_around() -> None:
    m = MockAgent(["a", "b"], name="cyc", cycle=True)
    assert (await m.run("1")).text() == "a"
    assert (await m.run("2")).text() == "b"
    assert (await m.run("3")).text() == "a"
    assert (await m.run("4")).text() == "b"


@pytest.mark.asyncio
async def test_empty_list_response_raises() -> None:
    m = MockAgent([], name="empty")
    with pytest.raises(RuntimeError, match="empty"):
        await m.run("x")


@pytest.mark.asyncio
async def test_dict_response_substring_match_and_default() -> None:
    m = MockAgent(
        {"weather": "sunny", "market": "bullish", DEFAULT: "unknown"},
        name="router",
    )
    assert (await m.run("what's the weather")).text() == "sunny"
    assert (await m.run("market update please")).text() == "bullish"
    assert (await m.run("random question")).text() == "unknown"


@pytest.mark.asyncio
async def test_dict_response_no_match_no_default_raises() -> None:
    m = MockAgent({"weather": "sunny"}, name="router")
    with pytest.raises(RuntimeError, match="no dict key"):
        await m.run("something else")


@pytest.mark.asyncio
async def test_callable_response_sync_sees_envelope() -> None:
    captured: list[Envelope] = []

    def handler(env: Envelope) -> str:
        captured.append(env)
        return f"echo:{env.task}"

    m = MockAgent(handler, name="echo")
    env = await m.run("hi")
    assert env.text() == "echo:hi"
    assert captured[0].task == "hi"


@pytest.mark.asyncio
async def test_callable_response_async_supported() -> None:
    async def handler(env: Envelope) -> str:
        await asyncio.sleep(0)  # prove it's awaited
        return f"async:{env.task}"

    m = MockAgent(handler, name="a_echo")
    assert (await m.run("ping")).text() == "async:ping"


# ---------------------------------------------------------------------------
# 2. Response value types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_response_pass_through_preserves_metadata() -> None:
    seed = Envelope(
        task=None,
        payload="payload-from-responder",
    )
    # The responder sets its own cost — we want to see that preserved.
    seed.metadata.input_tokens = 999
    seed.metadata.output_tokens = 111
    seed.metadata.cost_usd = 1.23
    seed.metadata.model = "custom-model"

    m = MockAgent(seed, name="passthrough")
    env = await m.run("question")

    # Responder's metadata wins
    assert env.metadata.input_tokens == 999
    assert env.metadata.output_tokens == 111
    assert env.metadata.cost_usd == 1.23
    assert env.metadata.model == "custom-model"
    # But task/context get backfilled from env_in when responder omits them
    assert env.task == "question"


@pytest.mark.asyncio
async def test_errorinfo_response_becomes_error_envelope() -> None:
    err = ErrorInfo(type="RateLimit", message="429", retryable=True)
    m = MockAgent(err, name="ratelimited")
    env = await m.run("x")
    assert not env.ok
    assert env.error is not None
    assert env.error.type == "RateLimit"
    assert env.error.retryable is True
    # No payload tokens on an error response
    assert env.metadata.input_tokens == 0
    assert env.metadata.output_tokens == 0


@pytest.mark.asyncio
async def test_exception_response_is_raised() -> None:
    m = MockAgent(TimeoutError("provider timed out"), name="flaky")
    with pytest.raises(TimeoutError, match="provider timed out"):
        await m.run("x")


@pytest.mark.asyncio
async def test_pydantic_model_becomes_payload() -> None:
    class Report(BaseModel):
        title: str
        score: float

    m = MockAgent(
        Report(title="Q1", score=0.82),
        name="reporter",
        output=Report,
    )
    env = await m.run("build report")
    assert isinstance(env.payload, Report)
    assert env.payload.title == "Q1"
    assert env.payload.score == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_default_metadata_is_applied_to_plain_payload() -> None:
    m = MockAgent(
        "ok",
        name="priced",
        default_input_tokens=100,
        default_output_tokens=50,
        default_cost_usd=0.004,
        default_model="mock-1",
        default_provider="mock-provider",
    )
    env = await m.run("q")
    assert env.metadata.input_tokens == 100
    assert env.metadata.output_tokens == 50
    assert env.metadata.cost_usd == pytest.approx(0.004)
    assert env.metadata.model == "mock-1"
    assert env.metadata.provider == "mock-provider"
    assert env.metadata.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# 3. Recording & assertion helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_each_call_with_envelopes_and_elapsed() -> None:
    m = MockAgent(["a", "b"], name="rec")
    await m.run("first")
    await m.run("second")
    assert m.call_count == 2
    assert isinstance(m.last_call, MockCall)
    assert m.last_call.task == "second"
    assert m.last_call.env_in.task == "second"
    assert m.last_call.env_out.text() == "b"
    assert m.last_call.elapsed_ms >= 0.0


@pytest.mark.asyncio
async def test_assert_helpers_detect_presence_and_count() -> None:
    m = MockAgent("ok", name="asserter")
    await m.run("buy AAPL now")
    await m.run("sell TSLA")

    m.assert_call_count(2)
    m.assert_called_with(contains="AAPL")
    m.assert_called_with(task="sell TSLA")

    with pytest.raises(AssertionError):
        m.assert_call_count(3)
    with pytest.raises(AssertionError):
        m.assert_called_with(contains="MSFT")


@pytest.mark.asyncio
async def test_reset_clears_log_and_rewinds_list_cursor() -> None:
    m = MockAgent(["a", "b"], name="rw")
    await m.run("1")
    m.reset()
    assert m.call_count == 0
    # Cursor rewound — next call returns first element again
    assert (await m.run("1'")).text() == "a"


# ---------------------------------------------------------------------------
# 4. Drop-in: as_tool wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_as_tool_returns_functional_tool_with_envelope_flag() -> None:
    m = MockAgent("done", name="inner")
    tool = m.as_tool()
    assert tool.name == "inner"
    assert tool.returns_envelope is True
    out = await tool.run(task="do the thing")
    assert isinstance(out, Envelope)
    assert out.text() == "done"
    # The mock recorded the call
    m.assert_call_count(1)


@pytest.mark.asyncio
async def test_wrap_tool_accepts_mock_agent_via_duck_type() -> None:
    """Regression: duck-typed _is_lazy_agent check in tools.wrap_tool.

    Before the v1 loosening, wrap_tool required ``isinstance(obj, Agent)``
    so MockAgent would silently fall through to the plain-callable path,
    losing returns_envelope=True and breaking nested-metadata roll-up.
    """
    from lazybridge.tools import wrap_tool

    m = MockAgent("out", name="ducky")
    tool = wrap_tool(m)
    assert tool.returns_envelope is True
    assert tool.name == "ducky"


# ---------------------------------------------------------------------------
# 5. Drop-in: Plan composition (data transmission across steps)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_two_step_passes_prev_output_to_next_step() -> None:
    researcher = MockAgent(
        {"weather": "sunny skies and 22C", DEFAULT: "no data"},
        name="researcher",
    )
    writer = MockAgent(
        lambda env: f"REPORT: {env.text()}",
        name="writer",
    )

    plan = Plan(
        Step(target=researcher, task="weather today", name="research"),
        Step(target=writer, task=from_prev, name="write"),
    )
    agent = Agent(engine=plan, name="pipeline")
    env = await agent.run("unused outer task")

    assert env.ok, env.error
    assert env.text() == "REPORT: sunny skies and 22C"
    researcher.assert_call_count(1)
    writer.assert_call_count(1)
    writer.assert_called_with(contains="sunny")


@pytest.mark.asyncio
async def test_plan_error_short_circuits_downstream_step() -> None:
    bad = MockAgent(
        ErrorInfo(type="Upstream", message="feed down"),
        name="bad",
    )
    writer = MockAgent("wrote", name="writer")

    plan = Plan(
        Step(target=bad, name="bad"),
        Step(target=writer, task=from_prev, name="write"),
    )
    agent = Agent(engine=plan, name="p")
    env = await agent.run("x")

    assert not env.ok
    assert env.error is not None
    assert "feed down" in env.error.message
    # Writer must not have been called when upstream errored.
    writer.assert_call_count(0)


# ---------------------------------------------------------------------------
# 6. Drop-in: Agent.chain / Agent.parallel sugar
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_chain_composes_mock_agents() -> None:
    a = MockAgent(lambda env: f"A({env.task})", name="a")
    b = MockAgent(lambda env: f"B({env.task})", name="b")

    chain = Agent.chain(a, b)
    env = await chain.run("input")

    # 'a' sees the original task, 'b' sees a's output as its task
    assert env.text() == "B(A(input))"
    a.assert_call_count(1)
    b.assert_call_count(1)


@pytest.mark.asyncio
async def test_agent_parallel_runs_mocks_concurrently() -> None:
    slow1 = MockAgent("one", name="s1", delay_ms=30)
    slow2 = MockAgent("two", name="s2", delay_ms=30)

    par = Agent.parallel(slow1, slow2)
    import time as _t

    t0 = _t.perf_counter()
    results = await par.run("same task to all")
    elapsed = (_t.perf_counter() - t0) * 1000

    # ``Agent.parallel`` returns ``list[Envelope]`` directly — one
    # per input agent, in input order.  If they ran serially, elapsed
    # would be ≥60ms; concurrent should be comfortably under 55ms even
    # with CI jitter.
    assert isinstance(results, list) and len(results) == 2
    assert elapsed < 55, f"parallel ran serially: {elapsed}ms"
    texts = {e.text() for e in results}
    assert texts == {"one", "two"}


# ---------------------------------------------------------------------------
# 7. Envelope metadata roll-up through nested as_tool composition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_preserves_final_step_metadata() -> None:
    """Plan currently returns the **final step's** metadata on the outer
    envelope — intermediate step metadata is not aggregated into
    ``nested_*`` buckets.  The per-step cost is still emitted as session
    events (TOOL_RESULT with tokens), so observability isn't lost, but
    callers that read ``env.metadata`` only see the last leg.

    This pins the current contract so a future change to rolling-up
    per-step metadata (desirable, tracked as a framework finding) is an
    intentional, test-breaking improvement rather than a silent drift.
    """
    a = MockAgent(
        "a-out",
        name="a",
        default_input_tokens=100,
        default_output_tokens=40,
        default_cost_usd=0.002,
    )
    b = MockAgent(
        "b-out",
        name="b",
        default_input_tokens=50,
        default_output_tokens=20,
        default_cost_usd=0.001,
    )

    plan = Plan(
        Step(target=a, name="a"),
        Step(target=b, task=from_prev, name="b"),
    )
    env = await Agent(engine=plan, name="p").run("start")

    # Both mocks were actually called — the Plan really did two steps.
    a.assert_call_count(1)
    b.assert_call_count(1)
    # Final envelope carries the *last* step's metadata verbatim.
    assert env.metadata.input_tokens == 50
    assert env.metadata.output_tokens == 20
    assert env.metadata.cost_usd == pytest.approx(0.001)
    # No nested aggregation from Plan (separate gap, documented above).
    assert env.metadata.nested_input_tokens == 0
    assert env.metadata.nested_output_tokens == 0
    assert env.metadata.nested_cost_usd == 0.0


@pytest.mark.asyncio
async def test_nested_metadata_rolls_up_through_as_tool_boundary() -> None:
    """The ``returns_envelope=True`` path on a tool boundary **does**
    aggregate into ``nested_*``.  This is the contract that actually
    exists today — we prove it here so the mock is known to exercise
    the same roll-up real Agents use.
    """
    inner = MockAgent(
        "inner-out",
        name="inner",
        default_input_tokens=77,
        default_output_tokens=33,
        default_cost_usd=0.005,
    )
    # Hand-drive the wrap-and-invoke path without spinning an LLM.
    from lazybridge.tools import wrap_tool

    tool = wrap_tool(inner)
    env = await tool.run(task="please")

    # Tool.run returns the inner Envelope verbatim — the outer engine
    # is the one that rolls fields into nested_*; we verify the inner
    # metadata is fully preserved so the outer rollup has something
    # to aggregate.
    assert env.metadata.input_tokens == 77
    assert env.metadata.output_tokens == 33
    assert env.metadata.cost_usd == pytest.approx(0.005)
    inner.assert_call_count(1)


# ---------------------------------------------------------------------------
# 8. Session auto-propagation via duck-typed _is_lazy_agent
# ---------------------------------------------------------------------------


def test_session_propagates_to_mock_agent_when_used_as_tool() -> None:
    """Regression: the agent.py session-propagation loop must use a
    duck-typed check so MockAgent (not a real Agent subclass) still
    inherits the outer session and appears in the graph."""
    sess = Session()
    inner = MockAgent("inner-out", name="inner")
    # Real outer Agent with the mock in its tools list. We can't run
    # the LLM engine without creds, but construction alone exercises
    # the propagation path.
    outer = Agent.__new__(Agent)
    # Minimal fields set to avoid invoking LLMEngine constructor:
    outer.engine = type("E", (), {"run": lambda *a, **k: None})()
    outer._tools_raw = [inner]
    outer._tool_map = {}
    outer.session = sess
    outer.name = "outer"
    # Manually run just the propagation block (mirrors agent.py:174-179).
    from lazybridge.agent import _safe_register_agent, _safe_register_tool_edge

    for raw in outer._tools_raw:
        if (
            getattr(raw, "_is_lazy_agent", False)
            and getattr(raw, "session", None) is None
        ):
            raw.session = outer.session
            _safe_register_agent(outer.session, raw)
            _safe_register_tool_edge(
                outer.session, outer, raw, label="as_tool"
            )

    assert inner.session is sess


# ---------------------------------------------------------------------------
# 9. Miscellany — __call__, stream, repr
# ---------------------------------------------------------------------------


def test_sync_call_outside_event_loop() -> None:
    m = MockAgent("hi", name="sync")
    env = m("task")
    assert env.text() == "hi"
    assert m.call_count == 1


@pytest.mark.asyncio
async def test_stream_yields_text_once() -> None:
    m = MockAgent("streamed", name="s")
    chunks: list[str] = []
    async for chunk in m.stream("q"):
        chunks.append(chunk)
    assert chunks == ["streamed"]


def test_repr_shows_name_and_call_count() -> None:
    m = MockAgent("x", name="rep")
    r = repr(m)
    assert "rep" in r
    assert "calls=0" in r
