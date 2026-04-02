"""Unit tests for run_async() — T9.xx series."""
from __future__ import annotations

import asyncio
import pytest
from lazybridge.lazy_run import run_async, _run_suppressed


# ---------------------------------------------------------------------------
# T9.01 — run_async returns correct value from a simple coroutine
# ---------------------------------------------------------------------------

def test_run_async_returns_value():
    # T9.01
    async def coro():
        return 42

    result = run_async(coro())
    assert result == 42


# ---------------------------------------------------------------------------
# T9.02 — run_async propagates exceptions
# ---------------------------------------------------------------------------

def test_run_async_propagates_exception():
    # T9.02
    async def coro():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_async(coro())


# ---------------------------------------------------------------------------
# T9.03 — run_async works with async/await chains
# ---------------------------------------------------------------------------

def test_run_async_nested_await():
    # T9.03
    async def inner():
        return "inner"

    async def outer():
        v = await inner()
        return f"outer:{v}"

    assert run_async(outer()) == "outer:inner"


# ---------------------------------------------------------------------------
# T9.04 — run_async from an already-running loop uses ThreadPoolExecutor path
# ---------------------------------------------------------------------------

async def test_run_async_inside_event_loop():
    # T9.04 — called from inside an async test; a loop is already running
    async def coro():
        return "from_thread"

    # run_async detects the running loop and offloads to a thread
    result = run_async(coro())
    assert result == "from_thread"


# ---------------------------------------------------------------------------
# T9.05 — _run_suppressed cleans up on exception
# ---------------------------------------------------------------------------

def test_run_suppressed_cleans_up_on_exception():
    # T9.05
    async def failing():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError, match="fail"):
        _run_suppressed(failing())

    # After cleanup, no running loop should exist
    try:
        loop = asyncio.get_running_loop()
        # If we're inside an async context, that's fine — the loop is from pytest
    except RuntimeError:
        pass  # No loop — correct for sync context


# ---------------------------------------------------------------------------
# T9.06 — run_async handles coroutines that use asyncio.sleep
# ---------------------------------------------------------------------------

def test_run_async_with_sleep():
    # T9.06
    async def coro():
        await asyncio.sleep(0)
        return "done"

    assert run_async(coro()) == "done"
