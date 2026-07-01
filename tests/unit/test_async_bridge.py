"""Tests for the shared sync↔async bridge (``lazybridge._asyncbridge``).

Covers the semantics that used to diverge across the six call sites this
module consolidates (Agent / ParallelAgent / Tool.run_sync / Memory /
MockAgent / Plan.run_many):

* dispatch by loop state (no loop / nest_asyncio / real running loop),
* contextvars propagation in BOTH the worker and the nest_asyncio branch,
* ``timeout`` that actually cancels the coroutine (not just raises),
* exception type preservation,
* no "coroutine was never awaited" leak.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import threading
import warnings

import pytest

from lazybridge import _asyncbridge
from lazybridge._asyncbridge import run_coroutine_blocking

# Module-level ContextVar used to prove propagation across the boundary.
CVAR: contextvars.ContextVar[str] = contextvars.ContextVar("lb_test_cvar", default="unset")


# ---------------------------------------------------------------------------
# 1. Baseline — sync context
# ---------------------------------------------------------------------------


def test_sync_context_returns_value() -> None:
    async def _c() -> int:
        return 42

    assert run_coroutine_blocking(lambda: _c()) == 42


def test_sync_context_propagates_contextvar() -> None:
    token = CVAR.set("sync-value")
    try:

        async def _c() -> str:
            return CVAR.get()

        assert run_coroutine_blocking(lambda: _c()) == "sync-value"
    finally:
        CVAR.reset(token)


# ---------------------------------------------------------------------------
# 2 + 4. Real running loop, not patched → worker thread, contextvars flow in
# ---------------------------------------------------------------------------


async def test_worker_branch_uses_different_thread_and_sees_contextvar() -> None:
    caller_tid = threading.get_ident()
    seen: dict[str, object] = {}
    token = CVAR.set("worker-value")
    try:

        async def _probe() -> str:
            seen["tid"] = threading.get_ident()
            seen["cvar"] = CVAR.get()
            return "ok"

        # Called from inside a running (non-patched) loop → worker path.
        result = run_coroutine_blocking(lambda: _probe())
    finally:
        CVAR.reset(token)

    assert result == "ok"
    assert seen["tid"] != caller_tid  # ran on a worker thread
    assert seen["cvar"] == "worker-value"  # contextvars copied into the worker


# ---------------------------------------------------------------------------
# 3 + 5. nest_asyncio branch → in-loop, same thread, contextvars visible
# ---------------------------------------------------------------------------


class _FakeNestLoop:
    """Stand-in for a nest_asyncio-patched running loop.

    We can't set ``_nest_patched`` on a genuinely-running asyncio loop and
    then call ``run_until_complete`` on it (that raises "already running"
    without the real nest_asyncio monkeypatch).  This fake models exactly the
    contract the bridge relies on: ``_nest_patched=True`` + a
    ``run_until_complete`` that drives the coroutine on the *calling* thread.
    """

    _nest_patched = True

    def __init__(self) -> None:
        self.called = False

    def run_until_complete(self, coro: object) -> object:
        self.called = True
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)  # type: ignore[arg-type]
        finally:
            loop.close()


def test_nest_asyncio_branch_runs_in_loop_same_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    caller_tid = threading.get_ident()
    seen: dict[str, object] = {}
    fake = _FakeNestLoop()
    monkeypatch.setattr(_asyncbridge.asyncio, "get_running_loop", lambda: fake)

    token = CVAR.set("nest-value")
    try:

        async def _probe() -> str:
            seen["tid"] = threading.get_ident()
            seen["cvar"] = CVAR.get()
            return "ok"

        result = run_coroutine_blocking(lambda: _probe())
    finally:
        CVAR.reset(token)

    assert result == "ok"
    assert fake.called  # the in-loop branch was taken
    assert seen["tid"] == caller_tid  # ran on the caller's thread (no worker)
    assert seen["cvar"] == "nest-value"  # contextvars visible in the nest branch


# ---------------------------------------------------------------------------
# 6. timeout actually cancels the coroutine
# ---------------------------------------------------------------------------


def test_timeout_cancels_coroutine() -> None:
    state: dict[str, bool] = {}

    async def _slow() -> None:
        state["started"] = True
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        state["completed"] = True

    with pytest.raises(TimeoutError):
        run_coroutine_blocking(lambda: _slow(), timeout=0.05)

    assert state.get("started") is True
    assert state.get("cancelled") is True  # not merely a raised TimeoutError
    assert state.get("completed") is None  # the body never ran to the end


def test_no_timeout_completes() -> None:
    async def _quick() -> str:
        await asyncio.sleep(0.01)
        return "done"

    assert run_coroutine_blocking(lambda: _quick(), timeout=5.0) == "done"


# ---------------------------------------------------------------------------
# 7. exception type is preserved across the boundary
# ---------------------------------------------------------------------------


class _Boom(ValueError):
    pass


def test_exception_type_preserved() -> None:
    async def _boom() -> None:
        raise _Boom("kaboom")

    with pytest.raises(_Boom, match="kaboom"):
        run_coroutine_blocking(lambda: _boom())


# ---------------------------------------------------------------------------
# 8. no "coroutine was never awaited" warning (happy path + timeout path)
# ---------------------------------------------------------------------------


def test_no_coroutine_never_awaited_warning() -> None:
    async def _c() -> int:
        return 1

    async def _slow() -> None:
        await asyncio.sleep(5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_coroutine_blocking(lambda: _c())
        with pytest.raises(TimeoutError):
            run_coroutine_blocking(lambda: _slow(), timeout=0.02)
        gc.collect()

    leaked = [str(w.message) for w in caught if "never awaited" in str(w.message)]
    assert not leaked, leaked


# ---------------------------------------------------------------------------
# 9. asyncio.run shutdown semantics — async generators are finalised before
#    the loop closes (regression guard for the six unified call sites, three
#    of which previously called asyncio.run directly).
# ---------------------------------------------------------------------------


def test_async_generators_are_shutdown_before_loop_closes() -> None:
    state: dict[str, bool] = {}

    async def _gen() -> object:
        try:
            yield 1
            yield 2
        finally:
            # Runs only if loop.shutdown_asyncgens() drives aclose() on a
            # generator that was never fully consumed / explicitly closed.
            state["finally_ran"] = True

    async def _main() -> object:
        agen = _gen()
        await agen.__anext__()  # partially consume; do NOT close it
        return agen  # keep it referenced so GC doesn't finalise it early

    holder = run_coroutine_blocking(lambda: _main())
    assert state.get("finally_ran") is True
    del holder


# ---------------------------------------------------------------------------
# Integration — the unified call sites work from inside a running loop
# (this is the path that used to diverge / hang).
# ---------------------------------------------------------------------------


async def test_mockagent_sync_call_from_running_loop() -> None:
    from lazybridge.testing import MockAgent

    m = MockAgent("hi", name="m")
    env = m("task")  # MockAgent.__call__ dispatched through the bridge
    assert env.text() == "hi"


async def test_tool_run_sync_async_func_from_running_loop() -> None:
    from lazybridge.tools import Tool

    async def af(x: int) -> int:
        return x + 1

    t = Tool(af, name="af")
    assert t.run_sync(x=1) == 2


def test_tool_run_sync_sync_func_bypasses_bridge() -> None:
    from lazybridge.tools import Tool

    def sf(x: int) -> int:
        return x * 3

    t = Tool(sf, name="sf")
    assert t.run_sync(x=4) == 12
