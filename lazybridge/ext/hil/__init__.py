"""Human-in-the-loop engines and ergonomic factories.

Two engines that involve a human in the agent run-loop:

- :class:`lazybridge.ext.hil.HumanEngine` — lightweight approval gate.
  Pauses an agent at the engine boundary and waits for a yes / redirect
  decision before producing the final envelope.
- :class:`lazybridge.ext.hil.SupervisorEngine` — REPL-style human
  supervision with tool calling, agent retries, and store inspection.

HIL lives in ``ext`` rather than core because it's a workflow pattern,
not a primitive every agent needs.  See ``docs/guides/core-vs-ext.md``
for the policy.

Two equivalent ways to construct a HIL agent:

1. Engine + escape-hatch factory on Agent::

       from lazybridge import Agent
       from lazybridge.ext.hil import SupervisorEngine
       Agent.from_engine(SupervisorEngine(tools=[...], agents=[...]))

2. Module-level factory (symmetric with ``Agent.from_*`` core factories;
   accepts the same uniform Agent kwargs through ``**agent_kwargs``)::

       from lazybridge.ext.hil import supervisor_agent, human_agent
       supervisor_agent(tools=[search], agents=[researcher], session=sess)
       human_agent(timeout=60.0, default="approve")

Both paths return the same :class:`Agent` shape — the second is sugar
for the first plus the typical kwargs forwarding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lazybridge.ext.hil.human import HumanEngine
from lazybridge.ext.hil.supervisor import SupervisorEngine

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from lazybridge import Agent


def supervisor_agent(
    *,
    tools: list[Any] | None = None,
    agents: list[Any] | None = None,
    store: Any | None = None,
    input_fn: Callable[[str], str] | None = None,
    ainput_fn: Callable[[str], Awaitable[str]] | None = None,
    timeout: float | None = None,
    default: str | None = None,
    **agent_kwargs: Any,
) -> Agent:
    """Build a human-supervised :class:`Agent` (REPL + tool dispatch + retry).

    Symmetric counterpart of ``Agent.from_<kind>(...)`` for the
    :class:`SupervisorEngine`.  Kept on the ext side rather than as
    ``Agent.from_supervisor`` to respect the core/ext import boundary
    (see ``docs/guides/core-vs-ext.md``).

    Engine kwargs (``tools``, ``agents``, ``store``, ``input_fn`` /
    ``ainput_fn``, ``timeout``, ``default``) configure the
    :class:`SupervisorEngine`; remaining ``**agent_kwargs`` (``memory=`` /
    ``session=`` / ``output=`` / ``verify=`` / ``fallback=`` / ``guard=`` /
    ``name=`` / etc.) flow to the unified Agent constructor::

        from lazybridge.ext.hil import supervisor_agent

        supervisor_agent(
            tools=[search],
            agents=[researcher],   # human can `retry researcher: <feedback>`
            session=sess,
            name="ops-supervisor",
        )("publish a policy brief")
    """
    # Local import — ``Agent`` lives in core, but core never imports
    # from ext, only the reverse, so this is the architecturally
    # correct direction.
    from lazybridge import Agent

    engine = SupervisorEngine(
        tools=tools,
        agents=agents,
        store=store,
        input_fn=input_fn,
        ainput_fn=ainput_fn,
        timeout=timeout,
        default=default,
    )
    return Agent(engine=engine, **agent_kwargs)


def human_agent(
    *,
    timeout: float | None = None,
    ui: Literal["terminal", "web"] | Any = "terminal",
    default: str | None = None,
    **agent_kwargs: Any,
) -> Agent:
    """Build a human-input :class:`Agent` (approval gate / form-style HIL).

    Symmetric counterpart of ``Agent.from_<kind>(...)`` for the
    :class:`HumanEngine`.  Use this for **synchronous human input** —
    a prompt at the terminal or a web form — rather than the full REPL
    of :func:`supervisor_agent`.

    Engine kwargs (``timeout``, ``ui``, ``default``) configure the
    :class:`HumanEngine`; remaining ``**agent_kwargs`` flow to the
    unified Agent constructor::

        from lazybridge.ext.hil import human_agent

        human_agent(timeout=60.0, default="approve")("Approve deploy?")
    """
    from lazybridge import Agent

    engine = HumanEngine(timeout=timeout, ui=ui, default=default)
    return Agent(engine=engine, **agent_kwargs)


__all__ = [
    # Engines (compose via ``Agent.from_engine(...)``).
    "HumanEngine",
    "SupervisorEngine",
    # Module-level factories — symmetric with ``Agent.from_<kind>(...)``.
    "supervisor_agent",
    "human_agent",
]
