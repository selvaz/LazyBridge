"""Engine Protocol — the single abstraction all engines implement."""

from __future__ import annotations

import contextvars
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lazybridge.envelope import Envelope
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.tools import Tool

# ---------------------------------------------------------------------------
# Run identity — which Agent is driving the current engine invocation.
#
# Engines are shareable objects: two Agents may hold the same LLMEngine
# instance.  Stamping the agent's name onto the engine
# (``engine._agent_name = ...``) therefore attributes every event and
# usage row to whichever Agent was *constructed* last, not the one
# actually running.  The context variable below carries the identity
# per-invocation instead: ``Agent`` binds it around each ``engine.run()``
# / ``engine.stream()`` call, and engines read it via
# :func:`resolve_agent_name`.  The ``_agent_name`` attribute is kept as a
# fallback for code that drives an engine directly, without an Agent.
# ---------------------------------------------------------------------------

_CURRENT_AGENT_NAME: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "lazybridge_current_agent_name",
    default=None,
)


def bind_agent_name(name: str) -> contextvars.Token[str | None]:
    """Bind the running agent's name to the current context.

    Returns a token to pass to :func:`unbind_agent_name` (in a
    ``finally``) when the invocation completes.
    """
    return _CURRENT_AGENT_NAME.set(name)


def unbind_agent_name(token: contextvars.Token[str | None]) -> None:
    """Undo :func:`bind_agent_name`."""
    _CURRENT_AGENT_NAME.reset(token)


def resolve_agent_name(engine: Any, default: str) -> str:
    """Name to attribute the current engine invocation to.

    Priority: the context-bound name (set by the Agent driving this run)
    → the engine's ``_agent_name`` attribute (direct engine use) →
    ``default``.
    """
    bound = _CURRENT_AGENT_NAME.get()
    if bound is not None:
        return bound
    return getattr(engine, "_agent_name", default) or default


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
        env: Envelope[Any],
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Any | None = None,
        plan_state: Any | None = None,
    ) -> Envelope[Any]: ...

    async def stream(
        self,
        env: Envelope[Any],
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
    ) -> AsyncIterator[str]: ...
