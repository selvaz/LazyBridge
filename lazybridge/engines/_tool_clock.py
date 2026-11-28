"""Per-run application-tool time accounting for coding engines."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Iterator
from typing import TypeVar

_T = TypeVar("_T")


class ToolClock:
    """Exclude application tools from request time and additional waits from idle time.

    ``idle_clock`` also pauses for application tools, but can be paused on its
    own for waits that must still count against the overall request budget.
    """

    def __init__(self, *, _idle_only: bool = False) -> None:
        self._count = 0
        self._paused_at = 0.0
        self._paused_seconds = 0.0
        self.last_resume = 0.0
        self.changed = asyncio.Event()
        self.idle_clock = self if _idle_only else ToolClock(_idle_only=True)

    @property
    def active(self) -> bool:
        return self._count > 0

    @property
    def paused_seconds(self) -> float:
        return self._paused_seconds + (time.monotonic() - self._paused_at if self.active else 0.0)

    def now(self) -> float:
        """Monotonic runtime time, frozen while any application tool runs."""
        return (self._paused_at if self.active else time.monotonic()) - self._paused_seconds

    def _notify(self) -> None:
        self.changed.set()
        self.changed = asyncio.Event()

    def enter(self) -> None:
        if self.idle_clock is not self:
            self.idle_clock.enter()
        if not self.active:
            self._paused_at = time.monotonic()
        self._count += 1
        self._notify()

    def exit(self) -> None:
        if not self.active:
            raise RuntimeError("ToolClock.exit() without enter()")
        self._count -= 1
        if not self.active:
            self._paused_seconds += time.monotonic() - self._paused_at
            self.last_resume = self.now()
        self._notify()
        if self.idle_clock is not self:
            self.idle_clock.exit()

    @contextlib.contextmanager
    def tool(self) -> Iterator[None]:
        """Release the clock even if a tool raises or is cancelled."""
        self.enter()
        try:
            yield
        finally:
            self.exit()


async def wait_with_tool_clock(
    awaitable: Awaitable[_T],
    timeout: float | None,
    clock: ToolClock | None = None,
    *,
    idle: bool = False,
) -> _T:
    """Like ``wait_for``, excluding tool time; idle windows restart on resume.

    Clock transitions wake the deadline task. There is no polling while paused.
    A real timeout cancels and joins the operation through ``asyncio.wait_for``.
    """
    if clock is None or timeout is None:
        return await asyncio.wait_for(awaitable, timeout)

    if idle:
        clock = clock.idle_clock
    started = clock.now()

    async def deadline() -> None:
        while True:
            changed = clock.changed
            origin = max(started, clock.last_resume) if idle else started
            remaining = origin + timeout - clock.now()
            if not clock.active and remaining <= 0:
                return
            # Not ``wait_for``: before Python 3.12 it swallows a cancellation
            # that lands in the same loop tick the awaited event fires. The
            # timer would then loop forever and the ``await timer`` in the
            # ``finally`` below would hang the caller (seen live as a Codex
            # run that finished its turn and never returned). ``wait`` always
            # propagates cancellation; a timeout just returns and is rechecked
            # against the clock at the top of the loop.
            waiter = asyncio.ensure_future(changed.wait())
            try:
                await asyncio.wait({waiter}, timeout=None if clock.active else remaining)
            finally:
                waiter.cancel()

    task = asyncio.ensure_future(awaitable)
    timer = asyncio.create_task(deadline())
    try:
        done, _ = await asyncio.wait((task, timer), return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            return await task
        return await asyncio.wait_for(task, 0)
    except asyncio.CancelledError:
        # wait() does not propagate cancellation to its children; wait_for does.
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        raise
    finally:
        timer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await timer
