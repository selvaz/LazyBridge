import asyncio

import pytest

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
    assert 0.09 <= clock.paused_seconds < 0.18
    assert not clock.active


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
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not clock.active
