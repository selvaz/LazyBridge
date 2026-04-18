"""lazy_run — sync bridge for async LazyAgent methods.

Allows calling any ``async def`` method from synchronous code without writing
``asyncio.run()`` by hand, and works correctly inside environments that already
have a running event loop (Jupyter notebooks, pytest-asyncio).

Usage::

    from lazybridge import LazyAgent, run_async

    agent = LazyAgent("anthropic")
    resp  = run_async(agent.achat("hello"))
    print(resp.content)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_logger = logging.getLogger(__name__)


def _run_suppressed(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* in a fresh event loop, suppressing spurious cleanup noise.

    httpx's ``AsyncClient`` schedules ``aclose()`` tasks at GC time.  When those
    tasks fire after the loop has already been closed (the normal asyncio.run()
    teardown sequence), Python prints "Task exception was never retrieved:
    RuntimeError: Event loop is closed" to stderr — purely cosmetic, tests pass.

    This wrapper:
    1. Installs a custom exception handler that silences that specific error.
    2. Cancels pending tasks before close, giving them a chance to finish cleanly.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handler(lp: asyncio.AbstractEventLoop, ctx: dict) -> None:
        exc = ctx.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return  # suppress httpx / anyio cleanup noise
        lp.default_exception_handler(ctx)

    loop.set_exception_handler(_handler)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            # Cancel any remaining tasks (mirrors asyncio.run() internal teardown)
            pending = {t for t in asyncio.all_tasks(loop) if not t.done()}
            if pending:
                for t in pending:
                    t.cancel()
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from synchronous code.

    Handles two execution contexts:

    * **Normal script / CLI** (no running event loop): runs directly in a fresh
      loop via :func:`_run_suppressed`.
    * **Jupyter / pytest-asyncio / Spyder** (event loop already running):
      offloads to a background thread to avoid deadlocking the live loop.

    Parameters
    ----------
    coro:
        Any awaitable produced by an ``async def`` method, e.g.
        ``agent.achat("hello")`` or ``agent.aloop("task", tools=[...])``.

    Returns
    -------
    The return value of the coroutine.

    Examples
    --------
    Single call::

        resp = run_async(agent.achat("What is 2+2?"))
        print(resp.content)

    With streaming::

        async def collect():
            async for chunk in agent.achat("hello", stream=True):
                print(chunk.text, end="", flush=True)

        run_async(collect())
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running — standard script / CLI context.
        return _run_suppressed(coro)
    else:
        # A loop is already running (Jupyter, async test frameworks, Spyder, etc.).
        # Offload to a dedicated thread to avoid blocking/deadlocking the live loop.
        # If the caller is already inside async code, preferring `await coro`
        # directly is cheaper than going through the thread pool — log at DEBUG
        # so power users can spot the overhead (audit L2).
        _logger.debug(
            "run_async called from inside a running event loop; offloading to a "
            "thread pool. In async code, prefer `await coro` directly."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_suppressed, coro)
            return future.result()
