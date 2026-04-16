"""LazyContext — composable, lazily-evaluated context injection.

A LazyContext is a callable that returns a string when invoked. It can be
built from multiple sources and composed with ``+`` or ``LazyContext.merge()``.

This replaces the implicit AgentMemory injection. Context is now:
  - explicit (you see it in code)
  - composable (combine any sources)
  - testable (call ctx() anywhere to see what the agent will receive)
  - decoupled from execution (LazyContext knows nothing about LazyAgent)

Sources::

    LazyContext.from_text("you are an analyst")
    LazyContext.from_function(get_user_profile)          # called at execution time
    LazyContext.from_store(store, keys=["findings"])      # reads LazyStore
    LazyContext.from_agent(agent)                         # reads agent's last output

Composition::

    ctx = LazyContext.from_agent(researcher) + LazyContext.from_text("reply in Italian")
    ctx = LazyContext.merge(ctx1, ctx2, ctx3)

Usage with LazyAgent::

    writer = LazyAgent("openai", context=ctx)
    writer.chat("write the article")
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lazybridge.lazy_store import LazyStore

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LazyContext
# ---------------------------------------------------------------------------


class LazyContext:
    """A lazily-evaluated, composable context string.

    Call ``ctx()`` or ``ctx.build()`` to materialise the string.
    """

    def __init__(self, _sources: list[Callable[[], str]] | None = None) -> None:
        self._sources: list[Callable[[], str]] = _sources or []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> LazyContext:
        """Static text — included verbatim in the context."""
        ctx = cls()
        ctx._sources.append(lambda: text)
        return ctx

    @classmethod
    def from_function(cls, fn: Callable[[], str]) -> LazyContext:
        """Call ``fn()`` at execution time and include the result."""
        ctx = cls()
        ctx._sources.append(fn)
        return ctx

    @classmethod
    def from_store(
        cls,
        store: LazyStore,
        *,
        keys: list[str] | None = None,
        prefix: str = "[store context]",
    ) -> LazyContext:
        """Read from a LazyStore at execution time.

        If ``keys`` is given, only those keys are included.
        """

        def _read() -> str:
            text = store.to_text(keys=keys) if (keys or store.keys()) else ""
            if not text:
                return ""
            label = prefix or "[store context]"
            return f"{label}\n{text}"

        ctx = cls()
        ctx._sources.append(_read)
        return ctx

    @classmethod
    def from_agent(
        cls,
        agent: Any,
        *,
        prefix: str | None = None,
    ) -> LazyContext:
        """Read the last output of another LazyAgent at execution time.

        The agent must have been run before this context is materialised,
        otherwise an empty string is returned.
        """

        def _read() -> str:
            output = getattr(agent, "_last_output", None)
            agent_name = getattr(agent, "name", None) or repr(agent)
            if output is None:
                _logger.debug("Agent %r has not been run yet; context will be empty.", agent_name)
                return ""
            if output == "":
                _logger.debug("Agent %r was run but returned an empty output; context will be empty.", agent_name)
                return ""
            label = prefix or (f"[{agent_name} output]")
            return f"{label}\n{output}"

        ctx = cls()
        ctx._sources.append(_read)
        return ctx

    @classmethod
    def merge(cls, *contexts: LazyContext) -> LazyContext:
        """Combine multiple contexts into one.  Sources are evaluated in order."""
        merged = cls()
        for ctx in contexts:
            merged._sources.extend(ctx._sources)
        return merged

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def build(self) -> str:
        """Materialise all sources and join with double newlines."""
        parts = []
        for source in self._sources:
            try:
                result = source()
                if result and result.strip():
                    parts.append(result.strip())
            except Exception as exc:
                _logger.warning("Context source %r failed (skipped): %s", source, exc)
        return "\n\n".join(parts)

    def __call__(self) -> str:
        """Shorthand for ``build()``."""
        return self.build()

    # ------------------------------------------------------------------
    # Composition operators
    # ------------------------------------------------------------------

    def __add__(self, other: LazyContext) -> LazyContext:
        if not isinstance(other, LazyContext):
            return NotImplemented
        return LazyContext.merge(self, other)

    def __radd__(self, other: Any) -> LazyContext:
        if isinstance(other, LazyContext):
            return LazyContext.merge(other, self)
        return NotImplemented

    def __bool__(self) -> bool:
        """True if at least one source is registered."""
        return bool(self._sources)

    def __repr__(self) -> str:
        return f"LazyContext({len(self._sources)} source(s))"
