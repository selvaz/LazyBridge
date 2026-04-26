"""Human-in-the-loop engines — alpha.

Two engines that involve a human in the agent run-loop:

- :class:`lazybridge.ext.hil.HumanEngine` — lightweight approval gate.
  Pauses an agent at the engine boundary and waits for a yes / redirect
  decision before producing the final envelope.
- :class:`lazybridge.ext.hil.SupervisorEngine` — REPL-style human
  supervision with tool calling, agent retries, and store inspection.

HIL lives in ``ext`` rather than core because it's a workflow pattern,
not a primitive every agent needs.  See ``docs/guides/core-vs-ext.md``
for the policy.

Usage::

    from lazybridge import Agent
    from lazybridge.ext.hil import HumanEngine, SupervisorEngine

    agent = Agent(engine=HumanEngine(...))
"""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"

from lazybridge.ext.hil.human import HumanEngine
from lazybridge.ext.hil.supervisor import SupervisorEngine

__all__ = ["HumanEngine", "SupervisorEngine"]
