"""Control-flow signals — non-local exits that cross nested agent boundaries.

``conclude`` lets any agent in a nested tree end the *whole* task and hand its
answer straight back to the top-level caller, skipping every intermediate
level.  It is implemented as a :class:`BaseException` (not ``Exception``) so it
slips past the engine's ``except Exception`` tool-error handlers untouched; the
LLMEngine loop re-raises it explicitly after ``asyncio.gather`` (which would
otherwise capture it into the results list), and only :meth:`Agent.run` — the
top-level entry point — converts it back into a normal :class:`Envelope`.
"""

from __future__ import annotations


class ConcludeSignal(BaseException):
    """Raised by :func:`conclude` to end the whole task with ``message``."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def conclude(message: str) -> str:
    """End the whole task now and return ``message`` as the final answer.

    Use as a tool in multi-agent graphs: any agent — however deeply nested —
    can call ``conclude("…")`` to short-circuit the entire call chain and
    return its answer directly to the original caller.
    """
    raise ConcludeSignal(message)
