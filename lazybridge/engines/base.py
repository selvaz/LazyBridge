"""Engine Protocol — the single abstraction all engines implement."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

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
    """

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
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
