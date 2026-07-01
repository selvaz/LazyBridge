"""Internal sync↔async bridge — the one implementation of "run a coroutine to
completion from synchronous code, wherever the event loop happens to be".

Private module: **not** part of the public API.  It consolidates what used to
be six near-identical, subtly-divergent bridges:

* ``Agent.__call__`` / ``ParallelAgent.__call__`` (``lazybridge.agent``)
* ``Tool.run_sync`` (``lazybridge.tools``)
* ``Memory._drive_to_completion`` (``lazybridge.memory``)
* ``MockAgent.__call__`` (``lazybridge.testing``)
* ``Plan.run_many`` (``lazybridge.engines.plan``)

Each had drifted: only ``agent`` handled ``nest_asyncio`` (Jupyter/Spyder) and
suppressed the httpx/anyio "Event loop is closed" GC noise; ``memory`` was the
only one honouring a ``timeout``; ``tools``/``memory``/``testing`` skipped the
in-loop nest_asyncio branch entirely, so the *same* call took a worker-thread
path in a notebook while ``Agent.__call__`` ran in-loop — a real source of
intermittent, path-dependent bugs.  This module makes every synchronous entry
point cross the async boundary with identical semantics.

Factory, not coroutine
----------------------
The entry point takes a **zero-argument factory** (``lambda: agent.run(task)``)
rather than an already-created coroutine.  The coroutine is instantiated only
at the moment — and on the thread — where it will actually be awaited, so a
failure anywhere in the dispatch logic can never strand a coroutine
"created but never awaited" (a ``RuntimeWarning`` plus a leaked task).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


def _suppress_loop_closed(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Swallow 'Event loop is closed' noise from httpx/anyio cleanup tasks.

    When a fresh loop is closed after the coroutine finishes, the GC may later
    call ``AsyncClient.__del__`` which tries to schedule an ``aclose()`` on the
    now-closed loop.  The resulting ``RuntimeError`` is benign — the request
    already completed — but without this handler it prints a confusing
    traceback to stderr.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and str(exc) == "Event loop is closed":
        return
    loop.default_exception_handler(context)


def _run_on_new_loop(coro: Awaitable[T]) -> T:
    """Run *coro* to completion on a fresh event loop, then drain + close it.

    Replaces a bare ``asyncio.run()`` so the "Event loop is closed" cleanup
    noise is suppressed without hiding real errors, and so pending tasks are
    cancelled and awaited before the loop closes (correctness, not cosmetics —
    a detached task left on a closing loop can swallow exceptions).
    """
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(_suppress_loop_closed)
    try:
        return loop.run_until_complete(coro)  # type: ignore[arg-type]
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                for task in pending:
                    task.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as exc:
            loop.call_exception_handler(
                {
                    "message": "Error while draining pending tasks during loop shutdown",
                    "exception": exc,
                }
            )
        loop.close()


def _run_in_worker(make_coro: Callable[[], Awaitable[T]]) -> T:
    """Run ``make_coro()`` on a fresh loop in a worker thread.

    The caller's :mod:`contextvars` context is copied into the thread so
    contextvars set by the outer framework (OpenTelemetry spans, request ids,
    structured-logging context) flow into the coroutine instead of starting
    empty.  The coroutine is created *inside* the worker (via the factory) so
    it is born on the thread that will await it.
    """
    ctx = contextvars.copy_context()

    def _target() -> T:
        return _run_on_new_loop(make_coro())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, _target).result()


def run_coroutine_blocking(make_coro: Callable[[], Awaitable[T]], *, timeout: float | None = None) -> T:
    """Run the coroutine produced by ``make_coro`` to completion from sync code.

    Dispatch by current event-loop state:

    * **no running loop** → a fresh dedicated loop on this thread
      (drain + loop-closed suppression).
    * **running loop patched by nest_asyncio** (Jupyter/Spyder set
      ``loop._nest_patched``) → run in-loop via ``run_until_complete``; nesting
      is safe there and avoids an event-loop lifetime mismatch that would leak
      httpx transports bound to a loop that never closes.
    * **running loop, not patched** (FastAPI, asyncio tests) → a fresh loop on
      a worker thread, with the caller's contextvars copied in.  Never
      ``run_until_complete`` on someone else's live loop.

    ``timeout`` (seconds), when set, is applied with ``asyncio.wait_for``
    *inside* the executing loop — never as a wall-clock join on the worker
    thread — so on expiry the coroutine is actually cancelled rather than left
    running detached, and a :class:`TimeoutError` propagates to the caller.
    """

    def _make_driver() -> Awaitable[T]:
        async def _driver() -> T:
            coro = make_coro()
            if timeout is None:
                return await coro
            return await asyncio.wait_for(coro, timeout)

        return _driver()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to run directly on a fresh loop.
        return _run_on_new_loop(_make_driver())

    if getattr(loop, "_nest_patched", False):
        # nest_asyncio patched the running loop — execute in-loop.
        return loop.run_until_complete(_make_driver())  # type: ignore[arg-type]

    # A real running loop we don't own — hop to a worker thread.
    return _run_in_worker(_make_driver)
