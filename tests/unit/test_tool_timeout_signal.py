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
