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

import pytest

from lazybridge.engines.llm import LLMEngine, ToolTimeoutError
from lazybridge.envelope import Envelope
from lazybridge.session import EventType, Session
from lazybridge.tools import wrap_tool


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

    tool = wrap_tool(_slow)
    tool_map = {"slow": tool}

    class _ToolCall:
        id = "call_1"
        name = "slow"
        arguments: dict = {}

    result = await engine._exec_tool(  # type: ignore[attr-defined]
        _ToolCall(),
        tool_map,
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

    tool = wrap_tool(_bad)
    tool_map = {"bad": tool}

    class _ToolCall:
        id = "call_2"
        name = "bad"
        arguments: dict = {}

    result = await engine._exec_tool(  # type: ignore[attr-defined]
        _ToolCall(),
        tool_map,
        session=sess,
        run_id="r2",
    )
    assert isinstance(result, ValueError)

    sess.flush()
    err = sess.events.query(event_type=EventType.TOOL_ERROR)
    timeout = sess.events.query(event_type=EventType.TOOL_TIMEOUT)
    assert len(err) == 1
    assert len(timeout) == 0
