"""Regression test for raw sync callables inside a Plan ``Step``.

The Plan executor used to call non-coroutine ``Step.target`` callables
directly on the event loop in ``_execute_one()``.  Any blocking I/O or
CPU work inside such a target would stall every other coroutine on the
loop, including the Session emitter and concurrent agent rungs.

After the fix in ``lazybridge/engines/plan/_plan.py``, sync raw
callables are dispatched to the default executor via
``loop.run_in_executor(None, ctx.run, target, arg)``.  The
``contextvars.copy_context()`` wrap is part of the contract: any
context-local state the caller set (request ID, OpenTelemetry trace
context, etc.) must survive the offload — the executor thread's
default empty context silently drops them otherwise.

This file pins both contracts:

1. Sync targets must not stall the event loop (sentinel-tick test).
2. Sync targets must observe the caller's contextvars
   (request-id propagation test).
"""

from __future__ import annotations

import asyncio
import contextvars
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
        # Assign so CodeQL's py/ineffectual-statement doesn't flag the
        # bare await — the drain has real side effects (raises any
        # sentinel exception, prevents pending-task warnings).
        _ = await sentinel_task

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


# A module-level ContextVar so the captured context survives across
# the Plan → Step → executor hop without lexical capture.
_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("test_request_id", default="<unset>")


def _read_request_id_sync(_task: str) -> str:
    """Synchronous target that reads ``_REQUEST_ID`` from contextvars.

    If the executor wrapper drops contextvars, this returns the default
    sentinel ``<unset>`` and the assertion below catches the regression.
    """
    return _REQUEST_ID.get()


@pytest.mark.asyncio
async def test_sync_callable_observes_caller_contextvars() -> None:
    plan = Plan(Step(target=_read_request_id_sync, name="read_ctx"))
    agent = Agent(engine=plan, name="t")

    token = _REQUEST_ID.set("req-abc-123")
    try:
        env = await agent.run("payload")
    finally:
        _REQUEST_ID.reset(token)

    # The contextvars.copy_context().run wrap is what makes this work;
    # without it the executor's empty thread-context returns the
    # ContextVar default ("<unset>").
    assert "req-abc-123" in env.text()
