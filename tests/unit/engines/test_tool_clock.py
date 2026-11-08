import asyncio

import pytest

from lazybridge.engines import _tool_clock
from lazybridge.engines._tool_clock import ToolClock, wait_with_tool_clock


@pytest.mark.asyncio
async def test_tool_time_is_excluded_and_overlap_counts_once():
    clock = ToolClock()

    async def work():
        with clock.tool():
            await asyncio.sleep(0.05)
            with clock.tool():
                await asyncio.sleep(0.05)
        return "done"

    assert await wait_with_tool_clock(work(), 0.03, clock) == "done"
    assert clock.paused_seconds > 0
    assert not clock.active


def test_overlapping_intervals_count_once_and_resume_only_after_last_tool(monkeypatch):
    now = 10.0
    monkeypatch.setattr(_tool_clock.time, "monotonic", lambda: now)
    clock = ToolClock()
    clock.enter()
    now = 12.0
    clock.enter()
    now = 14.0
    clock.exit()
    assert clock.active
    assert clock.paused_seconds == 4.0
    assert clock.now() == 10.0
    assert clock.last_resume == 0.0
    now = 16.0
    clock.exit()
    assert not clock.active
    assert clock.paused_seconds == 6.0
    assert clock.last_resume == 10.0
    now = 18.0
    assert clock.now() == 12.0


@pytest.mark.asyncio
async def test_request_budget_accumulates_across_tools():
    clock = ToolClock()

    async def work():
        await asyncio.sleep(0.08)
        with clock.tool():
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.08)

    with pytest.raises(TimeoutError):
        await wait_with_tool_clock(work(), 0.12, clock)
    assert not clock.active


@pytest.mark.asyncio
async def test_idle_window_restarts_after_tool():
    clock = ToolClock()

    async def work():
        await asyncio.sleep(0.08)
        with clock.tool():
            await asyncio.sleep(0.2)
        await asyncio.sleep(0.08)
        return "done"

    assert await wait_with_tool_clock(work(), 0.12, clock, idle=True) == "done"


@pytest.mark.parametrize("cleanup", ["return", "raise"])
@pytest.mark.asyncio
async def test_timeout_preserves_wait_for_cancellation_cleanup_semantics(cleanup):
    async def work():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            if cleanup == "raise":
                raise ValueError("cleanup failed") from None
            return "cancel suppressed"

    if cleanup == "return":
        assert await wait_with_tool_clock(work(), 0.03, ToolClock()) == "cancel suppressed"
    else:
        with pytest.raises(ValueError, match="cleanup failed"):
            await wait_with_tool_clock(work(), 0.03, ToolClock())


@pytest.mark.asyncio
async def test_runtime_stall_cancels_operation():
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()

    with pytest.raises(TimeoutError, match=r"^$"):
        await wait_with_tool_clock(work(), 0.03, ToolClock())
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_tool_scope_released_on_exception_and_cancellation():
    clock = ToolClock()
    with pytest.raises(ValueError), clock.tool():
        raise ValueError("tool failed")
    assert not clock.active

    entered = asyncio.Event()

    async def work():
        with clock.tool():
            entered.set()
            await asyncio.sleep(10)

    task = asyncio.create_task(wait_with_tool_clock(work(), 0.03, clock))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not clock.active
