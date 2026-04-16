"""LazyRouter — conditional branching in agent pipelines.

A LazyRouter takes a result/context and routes execution to one of several
LazyAgent instances based on a condition function.

For simple cases a Python ``if`` is fine. Use LazyRouter when you need:
  - Branching to be explicit and serialisable in the graph schema
  - The GUI to visualise conditional edges (diamond nodes)
  - Runtime routing among more than two alternatives

Usage::

    router = LazyRouter(
        condition=lambda result: "writer" if result.score > 0.5 else "reviewer",
        routes={
            "writer":   writer_agent,
            "reviewer": reviewer_agent,
        },
        name="quality_check",
    )

    # In a pipeline:
    result = analyst.chat("analyse the data")
    next_agent = router.route(result.content)
    final = next_agent.chat("continue from: " + result.content)

    # Or async:
    next_agent = await router.aroute(result.content)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class LazyRouter:
    """Route execution to one of several agents based on a condition.

    ``condition`` receives whatever value you pass to ``route()`` and
    returns the key of the agent to run next.

    ``routes`` maps string keys → LazyAgent instances.
    """

    condition: Callable[[Any], str]
    routes: dict[str, Any]  # str → LazyAgent
    name: str = "router"
    default: str | None = None  # fallback key if condition returns unknown key

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, value: Any) -> Any:
        """Evaluate condition and return the chosen LazyAgent (sync).

        Raises ``KeyError`` if the returned key is not in ``routes`` and no
        ``default`` is set.
        """
        key = self.condition(value)
        return self._resolve(key)

    async def aroute(self, value: Any) -> Any:
        """Async version of route().  Awaits the condition if it is a coroutine."""
        if inspect.iscoroutinefunction(self.condition):
            key = await self.condition(value)
        else:
            key = self.condition(value)
        return self._resolve(key)

    def _resolve(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise TypeError(
                f"LazyRouter '{self.name}': condition must return a str key, got {type(key).__name__!r} ({key!r})."
            )
        if key in self.routes:
            return self.routes[key]
        if self.default is not None and self.default in self.routes:
            return self.routes[self.default]
        raise KeyError(
            f"LazyRouter '{self.name}': condition returned key '{key}' "
            f"which is not in routes {list(self.routes.keys())}."
        )

    # ------------------------------------------------------------------
    # Graph introspection (for GraphSchema)
    # ------------------------------------------------------------------

    @property
    def agent_names(self) -> list[str]:
        """Names of all routable agents."""
        return [getattr(a, "name", k) for k, a in self.routes.items()]

    def to_graph_node(self) -> dict:
        """Serialisable representation for GraphSchema."""
        return {
            "type": "router",
            "name": self.name,
            "routes": {k: getattr(a, "id", str(k)) for k, a in self.routes.items()},
            "default": self.default,
        }

    def __repr__(self) -> str:
        return f"LazyRouter(name={self.name!r}, routes={list(self.routes.keys())})"
