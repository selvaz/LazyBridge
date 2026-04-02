"""lazy_run — sync bridge for async LazyAgent methods.

Allows calling any ``async def`` method from synchronous code without writing
``asyncio.run()`` by hand, and works correctly inside environments that already
have a running event loop (Jupyter notebooks, pytest-asyncio).

Usage::

    from lazybridgeframework import LazyAgent, run_async

    agent = LazyAgent("anthropic")
    resp  = run_async(agent.achat("hello"))
    print(resp.content)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from synchronous code.

    Handles two execution contexts transparently:

    * **Normal script / CLI** (no running event loop): delegates directly to
      ``asyncio.run()``.
    * **Jupyter / pytest-asyncio** (event loop already running): offloads to a
      background thread to avoid deadlocking the live loop.

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
        return asyncio.run(coro)
    else:
        # A loop is already running (Jupyter, async test frameworks, etc.).
        # Offload to a dedicated thread to avoid blocking/deadlocking the live loop.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
