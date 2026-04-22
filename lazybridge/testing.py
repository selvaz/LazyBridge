"""Test utilities for LazyBridge — scripted inputs, fake engines, helpers.

The name is intentionally module-level (``lazybridge.testing``) so test
modules can import stable public helpers instead of each recreating
private ``_scripted(...)`` closures.  Nothing here is used at runtime;
import only from tests / examples.

Public surface::

    from lazybridge.testing import (
        scripted_inputs,     # sync input_fn over a list / iterable
        scripted_ainputs,    # async variant (for SupervisorEngine.ainput_fn)
    )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable


def scripted_inputs(lines: Iterable[str]) -> Callable[[str], str]:
    """Return a sync ``input_fn`` that yields ``lines`` in order.

    Drop-in replacement for ``input()`` in HIL engine tests — lets the
    suite run deterministically without ``input()`` hangs.

    Raises ``StopIteration`` (caught by most REPL loops) when the
    scripted list is exhausted.  Supply an over-long list if you're
    unsure how many prompts the engine will issue.

    Example::

        from lazybridge import Agent, SupervisorEngine
        from lazybridge.testing import scripted_inputs

        sup = Agent(
            engine=SupervisorEngine(
                tools=[my_tool],
                input_fn=scripted_inputs(["my_tool(hello)", "continue"]),
            ),
            name="sup",
        )
        env = sup("task")
    """
    it = iter(lines)

    def _fn(_prompt: str) -> str:
        return next(it)

    return _fn


def scripted_ainputs(lines: Iterable[str]) -> Callable[[str], Awaitable[str]]:
    """Async counterpart of :func:`scripted_inputs`.

    Use when you want the supervisor's event-loop-native REPL path
    (pass as ``ainput_fn=``) — necessary to test cancellation /
    timeout semantics that don't exercise through the thread-pool
    fallback.
    """
    it = iter(lines)

    async def _fn(_prompt: str) -> str:
        return next(it)

    return _fn
