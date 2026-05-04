"""Engine Protocol — the single abstraction all engines implement."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lazybridge.envelope import Envelope
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool


@runtime_checkable
class Engine(Protocol):
    """Contract every engine must satisfy.

    ``run`` is the primary entry point: receives an Envelope, produces an Envelope.
    ``stream`` is optional; engines that do not support streaming raise NotImplementedError.

    The optional ``store`` and ``plan_state`` kwargs are consumed by
    :class:`Plan.run` for checkpoint / resume; other engines accept and
    ignore them.
    """

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Any | None = None,
        plan_state: Any | None = None,
    ) -> Envelope: ...

    async def stream(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncIterator[str]: ...
