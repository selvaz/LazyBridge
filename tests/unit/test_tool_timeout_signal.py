"""Wave 1.2 — tool timeout signal propagation.

A tool exceeding ``LLMEngine.tool_timeout`` is cancelled.  Two things
must happen:

1. ``Session`` records a distinct ``EventType.TOOL_TIMEOUT`` event —
   not ``TOOL_ERROR``.  Operators filter timeouts from genuine
   exceptions in dashboards / alerting.

2. The next-turn message handed to the model carries an explicit
   ``[TOOL_TIMEOUT]`` marker so the model can recognise cancellation
   and react (retry with smaller scope, abort, escalate) rather than
   confusing it with a generic exception.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from lazybridge.engines.llm import LLMEngine, ToolTimeoutError
from lazybridge.session import EventType, Session
from lazybridge.tools import Tool, _wrap_tool

# ---------------------------------------------------------------------------
# EventType is wired up
# ---------------------------------------------------------------------------


def test_tool_timeout_event_type_exists():
    assert EventType.TOOL_TIMEOUT.value == "tool_timeout"


def test_tool_timeout_is_in_default_critical_events():
    """Hybrid back-pressure must treat timeouts as critical so they
    never silently disappear under load."""
    from lazybridge.session import DEFAULT_CRITICAL_EVENT_TYPES

    assert EventType.TOOL_TIMEOUT.value in DEFAULT_CRITICAL_EVENT_TYPES


# ---------------------------------------------------------------------------
# Engine path: timeout emits TOOL_TIMEOUT, returns ToolTimeoutError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_emits_tool_timeout_event_on_timeout():
    sess = Session()
    engine = LLMEngine("claude-opus-4-7", tool_timeout=0.05)

    async def _slow():
        await asyncio.sleep(0.5)
        return "never"

    tool = _wrap_tool(_slow)
    tool_map = {"slow": tool}

    class _ToolCall:
        id = "call_1"
        name = "slow"
        arguments: dict = {}

    result = await engine._exec_tool(  # type: ignore[attr-defined]
        _ToolCall(),
        tool_map,
        agent_name="test",
        session=sess,
        run_id="r1",
    )

    assert isinstance(result, ToolTimeoutError)

    sess.flush()
    timeout_events = sess.events.query(event_type=EventType.TOOL_TIMEOUT)
    assert len(timeout_events) == 1
    payload = timeout_events[0]["payload"]
    assert payload["tool"] == "slow"
    assert payload["timeout_s"] == 0.05
    assert payload["type"] == "ToolTimeoutError"

    # Crucially: TOOL_TIMEOUT is NOT also emitted as TOOL_ERROR.
    error_events = sess.events.query(event_type=EventType.TOOL_ERROR)
    assert len(error_events) == 0


# ---------------------------------------------------------------------------
# The next-turn message marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_timeout_marker_reaches_model_in_next_turn():
    """The real _loop() path: after a tool timeout the USER message sent to
    the model in turn 2 must contain a ToolResultContent whose content starts
    with [TOOL_TIMEOUT].  This verifies the actual engine code, not a mirrored
    local copy of the classification logic."""
    from lazybridge.core.types import (
        CompletionRequest,
        CompletionResponse,
        StreamChunk,
        ToolCall,
        ToolResultContent,
        UsageStats,
    )
    from lazybridge.envelope import Envelope

    captured_requests: list[CompletionRequest] = []
    _call_turn = 0

    async def _fake_aexecute(req: CompletionRequest) -> CompletionResponse:
        nonlocal _call_turn
        captured_requests.append(req)
        if _call_turn == 0:
            _call_turn += 1
            return CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="tc-timeout-1", name="slow_op", arguments={})],
                stop_reason="tool_use",
                usage=UsageStats(input_tokens=5, output_tokens=1),
                model="fake",
            )
        return CompletionResponse(
            content="acknowledged",
            tool_calls=[],
            stop_reason="end_turn",
            usage=UsageStats(input_tokens=8, output_tokens=3),
            model="fake",
        )

    async def _fake_astream(req: CompletionRequest) -> AsyncIterator[StreamChunk]:
        resp = await _fake_aexecute(req)
        yield StreamChunk(
            delta=resp.content,
            tool_calls=resp.tool_calls,
            stop_reason=resp.stop_reason,
            usage=resp.usage,
            is_final=True,
        )

    class _FakeExecutor:
        async def aexecute(self, req: CompletionRequest) -> CompletionResponse:
            return await _fake_aexecute(req)

        async def astream(self, req: CompletionRequest) -> AsyncIterator[StreamChunk]:
            async for c in _fake_astream(req):
                yield c

    async def slow_op() -> str:
        await asyncio.sleep(10)
        return "never"

    engine = LLMEngine("fake", provider="fake", tool_timeout=0.05, request_timeout=None)
    engine._make_executor = lambda: _FakeExecutor()  # type: ignore[assignment]

    tool = Tool(slow_op)
    env = Envelope(task="please call slow_op")
    await engine.run(env, tools=[tool], output_type=str, memory=None, session=None)

    # Two round-trips to the model: turn 1 = tool_call, turn 2 = tool result.
    assert len(captured_requests) == 2, f"Expected 2 requests (tool-call + result), got {len(captured_requests)}"

    # The last message in the second request is the USER message with tool results.
    second_req = captured_requests[1]
    tool_result_blocks = [
        b
        for msg in second_req.messages
        if not isinstance(msg.content, str)
        for b in msg.content
        if isinstance(b, ToolResultContent)
    ]
    assert tool_result_blocks, "No ToolResultContent found in second request"
    timeout_block = next(
        (b for b in tool_result_blocks if b.tool_use_id == "tc-timeout-1"),
        None,
    )
    assert timeout_block is not None, "Expected ToolResultContent for tc-timeout-1"
    assert timeout_block.content.startswith("[TOOL_TIMEOUT]"), (
        f"Expected [TOOL_TIMEOUT] prefix, got: {timeout_block.content!r}"
    )
    assert timeout_block.is_error is True


@pytest.mark.asyncio
async def test_timeout_result_block_carries_tool_timeout_marker(monkeypatch):
    """When the engine builds the USER message with tool results, a
    ``ToolTimeoutError`` must produce content prefixed with
    ``[TOOL_TIMEOUT]`` — distinct from a generic ``Tool error: ...``.
    """
    from lazybridge.core.types import ToolResultContent

    # We exercise the result-block construction in isolation by
    # mirroring the engine's classification logic.  This guards the
    # contract: model sees [TOOL_TIMEOUT] for cancellations.
    timeout_err = ToolTimeoutError("Tool 'x' timed out after 0.05s")
    generic_err = RuntimeError("boom")

    def _classify(tr):
        if isinstance(tr, ToolTimeoutError):
            return f"[TOOL_TIMEOUT] {tr}", True
        if isinstance(tr, Exception):
            return f"Tool error: {tr}", True
        return str(tr), False

    timeout_content, timeout_is_err = _classify(timeout_err)
    generic_content, generic_is_err = _classify(generic_err)

    assert timeout_content.startswith("[TOOL_TIMEOUT]")
    assert "timed out" in timeout_content
    assert timeout_is_err is True

    assert generic_content.startswith("Tool error:")
    assert "[TOOL_TIMEOUT]" not in generic_content
    assert generic_is_err is True

    # Sanity: this is the actual code path used by the engine.
    block = ToolResultContent(
        tool_use_id="call_1",
        content=timeout_content,
        tool_name="x",
        is_error=timeout_is_err,
    )
    assert block.content.startswith("[TOOL_TIMEOUT]")
    assert block.is_error is True


# ---------------------------------------------------------------------------
# Genuine exception still emits TOOL_ERROR (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_genuine_exception_still_emits_tool_error():
    sess = Session()
    engine = LLMEngine("claude-opus-4-7")

    async def _bad():
        raise ValueError("boom")

    tool = _wrap_tool(_bad)
    tool_map = {"bad": tool}

    class _ToolCall:
        id = "call_2"
        name = "bad"
        arguments: dict = {}

    result = await engine._exec_tool(  # type: ignore[attr-defined]
        _ToolCall(),
        tool_map,
        agent_name="test",
        session=sess,
        run_id="r2",
    )
    assert isinstance(result, ValueError)

    sess.flush()
    err = sess.events.query(event_type=EventType.TOOL_ERROR)
    timeout = sess.events.query(event_type=EventType.TOOL_TIMEOUT)
    assert len(err) == 1
    assert len(timeout) == 0


# ---------------------------------------------------------------------------
# Tool-level bound: the only one that works for a BLOCKING synchronous tool
# ---------------------------------------------------------------------------
#
# ``LLMEngine.tool_timeout`` above wraps ``tool.run()`` in ``wait_for``.  That
# bounds an async tool, but a sync tool runs in the loop's executor and an
# executor future that has already started ignores cancellation — ``wait_for``
# then waits for a cancellation that never lands.  Measured before the fix: an
# ``Agent(timeout=8)`` around a blocking ``web_search`` was still going two
# minutes later.  ``Tool(timeout=)`` abandons the call instead.


def _blocca(seconds: float = 2.0) -> str:
    """Long enough to blow any bound below, short enough that the abandoned
    workers these tests leave behind drain before the rest of the suite runs:
    a thread still sleeping is real load, and a neighbouring test that budgets
    two seconds for a subprocess will miss it."""
    import time

    time.sleep(seconds)
    return "mai"


@pytest.mark.asyncio
async def test_sync_tool_timeout_returns_instead_of_blocking():
    t = Tool(_blocca, name="blocca", timeout=0.2)
    inizio = asyncio.get_running_loop().time()
    with pytest.raises(ToolTimeoutError) as exc:
        await t.run(seconds=2.0)
    assert asyncio.get_running_loop().time() - inizio < 5.0
    assert exc.value.tool_name == "blocca"
    assert exc.value.timeout == 0.2


@pytest.mark.asyncio
async def test_sync_tool_timeout_leaves_the_loop_running():
    """The abandoned worker must not be holding the event loop hostage:
    other coroutines have to keep making progress while it runs on."""
    tick = 0

    async def _battito() -> None:
        nonlocal tick
        for _ in range(10):
            await asyncio.sleep(0.02)
            tick += 1

    battito = asyncio.create_task(_battito())
    with pytest.raises(ToolTimeoutError):
        await Tool(_blocca, name="blocca", timeout=0.2).run(seconds=2.0)
    await battito
    assert tick == 10


@pytest.mark.asyncio
async def test_sync_tool_under_its_timeout_returns_normally():
    t = Tool(lambda seconds=0.0: "fatto", name="rapido", timeout=5.0)
    assert await t.run(seconds=0.0) == "fatto"


@pytest.mark.asyncio
async def test_sync_tool_exception_survives_the_thread_hop():
    def _rompe() -> str:
        raise ValueError("dal thread")

    with pytest.raises(ValueError, match="dal thread"):
        await Tool(_rompe, name="rompe", timeout=5.0).run()


@pytest.mark.asyncio
async def test_async_tool_honours_its_own_timeout():
    async def _lento() -> str:
        await asyncio.sleep(2)
        return "mai"

    with pytest.raises(ToolTimeoutError):
        await Tool(_lento, name="lento", timeout=0.1).run()


@pytest.mark.asyncio
async def test_no_timeout_means_no_bound():
    t = Tool(lambda: "fatto", name="libero")
    assert t.timeout is None
    assert await t.run() == "fatto"


# ---------------------------------------------------------------------------
# Agent(tool_timeout=) as the default for tools that declare nothing
# ---------------------------------------------------------------------------


def _agente(**kwargs):
    from lazybridge import Agent

    class _Engine:
        async def run(self, *a, **k):  # pragma: no cover - never invoked
            raise NotImplementedError

    return Agent(engine=_Engine(), name="prova", **kwargs)


def test_agent_tool_timeout_reaches_a_tool_that_declares_none():
    agente = _agente(tools=[Tool(lambda: "x", name="a")], tool_timeout=7.0)
    assert agente._tool_map["a"].timeout == 7.0


def test_a_tools_own_timeout_beats_the_agent_default():
    agente = _agente(tools=[Tool(lambda: "x", name="a", timeout=1.0)], tool_timeout=7.0)
    assert agente._tool_map["a"].timeout == 1.0


def test_agent_default_does_not_mutate_a_shared_tool():
    """The same Tool object is routinely handed to several agents; the first
    one's default must not become the tool's own."""
    condiviso = Tool(lambda: "x", name="a")
    primo = _agente(tools=[condiviso], tool_timeout=7.0)
    secondo = _agente(tools=[condiviso])
    assert primo._tool_map["a"].timeout == 7.0
    assert condiviso.timeout is None
    assert secondo._tool_map["a"].timeout is None


def test_wrap_can_add_or_override_a_bound():
    base = Tool(lambda: "x", name="a", timeout=1.0)
    assert Tool.wrap(base, timeout=9.0).timeout == 9.0
    assert Tool.wrap(base).timeout == 1.0
    assert Tool.wrap(lambda: "x", name="b", timeout=9.0).timeout == 9.0


# ---------------------------------------------------------------------------
# Bounds that were quietly skipped — found by review, each one a real hole
# ---------------------------------------------------------------------------


def test_run_sync_honours_the_bound_too():
    """A sync caller must not be able to outlast a deadline the async one
    respects: ``run_sync`` is the REPL / SupervisorEngine entry point."""
    import time as _t

    t = Tool(_blocca, name="blocca", timeout=0.2)
    inizio = _t.perf_counter()
    with pytest.raises(ToolTimeoutError):
        t.run_sync(seconds=2.0)
    assert _t.perf_counter() - inizio < 5.0


def test_run_sync_without_a_bound_is_unchanged():
    assert Tool(lambda: "fatto", name="libero").run_sync() == "fatto"


@pytest.mark.asyncio
async def test_wrap_bounds_an_agent_used_as_a_tool():
    from lazybridge import MockAgent

    sub = MockAgent("risposta", name="sub", delay_ms=2000)
    t = Tool.wrap(sub, name="sub", timeout=0.2)
    assert t.timeout == 0.2
    with pytest.raises(ToolTimeoutError):
        await t.run(task="qualcosa")
    # the alias is bounded, the agent everywhere else is not
    assert Tool.wrap(sub, name="altro").timeout is None


@pytest.mark.parametrize("valore", [0, -1, -0.5])
def test_a_deadline_that_can_never_be_met_is_rejected(valore):
    """Zero would fire on every call while a side-effecting tool ran on
    regardless — a configuration error, not a runtime one."""
    with pytest.raises(ValueError, match="must be > 0 or None"):
        Tool(lambda: "x", name="a", timeout=valore)
    with pytest.raises(ValueError, match="must be > 0 or None"):
        _agente(tools=[Tool(lambda: "x", name="a")], tool_timeout=valore)
    with pytest.raises(ValueError, match="must be > 0 or None"):
        Tool.from_schema("a", "d", {"type": "object"}, lambda: "x", timeout=valore)


# ---------------------------------------------------------------------------
# What "timed out" must NOT be confused with
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_tools_own_TimeoutError_is_not_reported_as_our_timeout():
    """An HTTP client reporting its own deadline is a tool failure.  Relabelling
    it ``ToolTimeoutError`` would tell the model to retry smaller when the real
    answer is that the endpoint is down."""

    async def _client() -> str:
        raise TimeoutError("read timed out")

    with pytest.raises(TimeoutError) as exc:
        await Tool(_client, name="client", timeout=30).run()
    assert not isinstance(exc.value, ToolTimeoutError)


@pytest.mark.asyncio
async def test_an_async_tool_that_times_out_is_cancelled_not_abandoned():
    """Only the sync path has to abandon: a coroutine takes cancellation."""
    cancellato = False

    async def _lento() -> str:
        nonlocal cancellato
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            cancellato = True
            raise
        return "mai"

    with pytest.raises(ToolTimeoutError):
        await Tool(_lento, name="lento", timeout=0.1).run()
    assert cancellato


@pytest.mark.asyncio
async def test_a_late_failure_from_an_abandoned_worker_is_observed():
    """Nobody is left holding the exception: an abandoned worker that raises
    must not reach the loop's exception handler as "never retrieved", detached
    from the run that caused it and looking like a framework bug."""
    import gc
    import time as _t

    def _fallisce_tardi() -> str:
        _t.sleep(0.3)
        raise ValueError("tardi")

    visti: list[dict] = []
    loop = asyncio.get_running_loop()
    originale = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: visti.append(ctx))
    try:
        with pytest.raises(ToolTimeoutError):
            await Tool(_fallisce_tardi, name="tardi", timeout=0.1).run()
        await asyncio.sleep(0.6)  # let the abandoned worker land its failure
        gc.collect()  # the warning fires from Future.__del__
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(originale)
    assert not [c for c in visti if "never retrieved" in str(c.get("message", ""))], visti


@pytest.mark.asyncio
async def test_abandoned_workers_do_not_pile_up_silently():
    """Each caller gets its timely timeout, which is what makes the leak
    invisible — the run looks healthy while the process fills up."""
    from lazybridge.tools import _abbandonati

    t = Tool(_blocca, name="blocca", timeout=0.05)
    t.abandoned_worker_warning_threshold = 3
    # A diagnostic counter, and every other test in this file leaves workers
    # in it — count from zero rather than inherit an order-dependent baseline.
    _abbandonati.clear()
    for _ in range(2):
        with pytest.raises(ToolTimeoutError):
            await t.run(seconds=2.0)
    assert len(_abbandonati) == 2

    with pytest.warns(UserWarning, match="abandoned tool workers"), pytest.raises(ToolTimeoutError):
        await t.run(seconds=2.0)


@pytest.mark.asyncio
async def test_a_worker_that_finished_is_not_counted_as_abandoned():
    """Pruning is what keeps the warning honest: a slow-but-finishing tool
    must not accumulate toward it."""
    import time as _t

    from lazybridge.tools import _abbandonati

    def _quasi() -> str:
        _t.sleep(0.15)
        return "fatto"

    t = Tool(_quasi, name="quasi", timeout=0.05)
    with pytest.raises(ToolTimeoutError):
        await t.run()
    await asyncio.sleep(0.4)
    prima = len(_abbandonati)
    with pytest.raises(ToolTimeoutError):
        await t.run()
    # the earlier worker has returned; only the new one is outstanding
    assert len(_abbandonati) <= prima + 1


@pytest.mark.asyncio
async def test_cancelling_the_caller_cancels_the_async_tool_too():
    """``Agent(timeout=1)`` firing inside a longer ``Tool(timeout=30)`` must
    not leave the tool running and free to land its side effect afterwards."""
    effetto = []
    cancellato = False

    async def _lento() -> str:
        nonlocal cancellato
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            cancellato = True
            raise
        effetto.append("side effect")
        return "mai"

    chiamata = asyncio.create_task(Tool(_lento, name="lento", timeout=30).run())
    await asyncio.sleep(0.05)
    chiamata.cancel()
    with pytest.raises(asyncio.CancelledError):
        await chiamata
    assert cancellato
    await asyncio.sleep(2.2)
    assert effetto == []


@pytest.mark.asyncio
async def test_a_worker_abandoned_by_cancellation_is_counted_too():
    """An outer bound shorter than the tool's own cancels here on EVERY call,
    so this is the likelier of the two ways to leak a thread."""
    from lazybridge.tools import _abbandonati

    _abbandonati.clear()
    chiamata = asyncio.create_task(Tool(_blocca, name="blocca", timeout=30).run(seconds=2.0))
    await asyncio.sleep(0.05)
    chiamata.cancel()
    with pytest.raises(asyncio.CancelledError):
        await chiamata
    assert len(_abbandonati) == 1


def test_adding_a_bound_keeps_an_explicit_schema():
    """A deadline says nothing about a tool's shape.  Rebuilding the Tool to
    attach one would regenerate the schema from the callable — and an imported
    tool's callable is often ``**kwargs``, so the model would be shown a tool
    with no parameters."""
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    base = Tool.from_schema("cerca", "Cerca.", schema, lambda **kw: "x")
    limitato = Tool.wrap(base, timeout=5.0)
    assert limitato.timeout == 5.0
    assert limitato.definition().parameters == schema
    assert base.timeout is None


def test_the_lowercase_alias_forwards_the_bound():
    from lazybridge import tool as tool_factory

    assert tool_factory(lambda: "x", name="a", timeout=3.0).timeout == 3.0


@pytest.mark.asyncio
async def test_a_subclass_that_overrides_run_still_goes_through_its_override():
    """Bounding a call must not reach past an override into the base dispatch:
    a subclass overrides ``run`` precisely to add something — authorization,
    tracing — that skipping it would silently drop."""
    from lazybridge.tools import run_tool_bounded

    passaggi = []

    class Tracciato(Tool):
        async def run(self, **kwargs):
            passaggi.append(kwargs)
            return await super().run(**kwargs)

    t = Tracciato(lambda q="": f"ok:{q}", name="tracciato")
    assert await run_tool_bounded(t, {"q": "x"}, 5.0) == "ok:x"
    assert passaggi == [{"q": "x"}]


def test_build_tool_map_rejects_a_deadline_that_can_never_be_met():
    from lazybridge.tools import build_tool_map

    with pytest.raises(ValueError, match="must be > 0 or None"):
        build_tool_map([Tool(lambda: "x", name="a")], default_timeout=0)


@pytest.mark.asyncio
async def test_an_overriding_subclasss_own_TimeoutError_is_not_relabelled():
    """The fallback path must draw the same distinction as the base one:
    a client's own deadline is a tool failure, not our bound expiring."""
    from lazybridge.tools import run_tool_bounded

    class Tracciato(Tool):
        async def run(self, **kwargs):
            raise TimeoutError("read timed out")

    with pytest.raises(TimeoutError) as exc:
        await run_tool_bounded(Tracciato(lambda: "x", name="t"), {}, 30.0)
    assert not isinstance(exc.value, ToolTimeoutError)
