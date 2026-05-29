"""AgentPool — name-based routing for dynamic multi-agent graphs.

A pool is a thin registry exposed to agents as a single ``route`` tool.  It
solves two problems that plain ``tools=[other_agent]`` composition cannot:

* **Circular references.** Agents that delegate to each other can't all be
  passed into each other's ``tools=[...]`` at construction (the tool map is
  frozen in ``Agent.__init__``).  Agents reference the *pool* (which already
  exists); the pool references the agents (registered afterwards).

* **Bounded cycles.** Because routing can form loops (A → B → A → …), the pool
  tracks call depth via a :class:`~contextvars.ContextVar` and refuses to
  recurse past ``max_depth`` — turning a would-be ``RecursionError`` into a
  plain message that nudges the model to ``conclude``.

Mechanically ``route`` is an ordinary tool; the engine does not special-case
it.  Pair it with :func:`lazybridge.conclude` so any agent can end the chain.

Example::

    pool = AgentPool()
    alice = Agent(name="alice", engine=..., tools=[pool.as_tool(), conclude])
    bob   = Agent(name="bob",   engine=..., tools=[pool.as_tool(), conclude])
    pool.register(alice, bob)
    result = alice.run("...")   # alice may route("bob", ...) and bob may conclude
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

from lazybridge.tools import Tool

if TYPE_CHECKING:
    from lazybridge.agent import Agent


class AgentPool:
    """Registry of named agents, exposed to the LLM as a single ``route`` tool."""

    def __init__(self, *, max_depth: int = 25) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth!r}")
        self._agents: dict[str, Agent] = {}
        self._max_depth = max_depth
        self._depth: contextvars.ContextVar[int] = contextvars.ContextVar(
            "lb_pool_depth", default=0
        )

    def register(self, *agents: Agent) -> None:
        """Add agents to the pool, keyed by their ``name``."""
        for agent in agents:
            if not getattr(agent, "name", None):
                raise ValueError("AgentPool.register() requires agents with an explicit name=.")
            self._agents[agent.name] = agent

    def roster(self) -> str:
        """One line per registered agent — drop into the members' system prompt."""
        return "\n".join(f"- {name}: {a.description or ''}" for name, a in self._agents.items())

    async def route(self, agent_name: str, task: str) -> str:
        """Delegate ``task`` to the named specialist agent and return its answer."""
        depth = self._depth.get()
        if depth >= self._max_depth:
            return f"Max routing depth {self._max_depth} reached — call conclude now with your best answer."
        agent = self._agents.get(agent_name)
        if agent is None:
            return f"Unknown agent {agent_name!r}. Available: {list(self._agents)}"
        token = self._depth.set(depth + 1)
        try:
            # ``_run_as_tool`` so a ``conclude`` raised downstream propagates
            # past this level to the original top-level caller.
            return (await agent._run_as_tool(task)).text()
        finally:
            self._depth.reset(token)

    def as_tool(self) -> Tool:
        """Wrap routing as a ``route(agent_name, task)`` tool for ``tools=[...]``."""
        return Tool.wrap(self.route, name="route")
