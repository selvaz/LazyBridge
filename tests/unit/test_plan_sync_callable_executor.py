"""Regression test for raw sync callables inside a Plan ``Step``.

The Plan executor used to call non-coroutine ``Step.target`` callables
directly on the event loop in ``_execute_one()``.  Any blocking I/O or
CPU work inside such a target would stall every other coroutine on the
loop, including the Session emitter and concurrent agent rungs.

After the fix in ``lazybridge/engines/plan/_plan.py``, sync raw
callables are dispatched to the default executor via
``loop.run_in_executor(None, target, arg)``.  This test pins that
contract by running a Plan step whose target sleeps synchronously for
a short interval while a sentinel coroutine ticks on the same loop —
the sentinel must keep progressing during the sleep.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from lazybridge import Agent, Step
from lazybridge.engines.plan import Plan


def _blocking_sync(task: str) -> str:
    """Synchronous target that blocks for 200ms via ``time.sleep``.

    ``time.sleep`` is the canonical "blocks the event loop" primitive;
    if the executor wiring regresses, the loop is frozen for the entire
    duration and the sentinel below will not tick.
    """
    time.sleep(0.2)
    return f"slept-for:{task}"


@pytest.mark.asyncio
async def test_sync_raw_callable_does_not_block_event_loop() -> None:
    plan = Plan(Step(target=_blocking_sync, name="blocker"))
    agent = Agent(engine=plan, name="t")

    ticks = 0
    stop = asyncio.Event()

    async def sentinel() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.02)

    sentinel_task = asyncio.create_task(sentinel())
    try:
        env = await agent.run("payload")
    finally:
        stop.set()
        await sentinel_task

    assert "slept-for:payload" in env.text()
    # The blocking sync target sleeps 200ms; the sentinel ticks every
    # 20ms.  If the loop is wedged, ticks stays at 0 / 1 — set the
    # floor at >= 3 ticks to leave headroom for slow CI runners while
    # still catching a wholesale regression.
    assert ticks >= 3, f"event loop stalled during sync target ({ticks} ticks)"


@pytest.mark.asyncio
async def test_async_raw_callable_still_awaited_natively() -> None:
    """The executor wiring must not break the coroutine-target path."""

    async def _async_target(task: str) -> str:
        await asyncio.sleep(0.01)
        return f"async:{task}"

    plan = Plan(Step(target=_async_target, name="async_step"))
    agent = Agent(engine=plan, name="t")
    env = await agent.run("payload")
    assert "async:payload" in env.text()
