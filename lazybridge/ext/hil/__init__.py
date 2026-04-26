"""Human-in-the-loop engines — alpha.

This extension hosts the two LazyBridge engines that involve a human in
the agent run-loop:

- :class:`lazybridge.ext.hil.HumanEngine` — lightweight approval gate.
  Pauses an agent at the engine boundary and waits for a yes / redirect
  decision before producing the final envelope.
- :class:`lazybridge.ext.hil.SupervisorEngine` — REPL-style human
  supervision with tool calling, agent retries, and store inspection.

Both engines were in ``lazybridge.engines`` before the core/ext split
formalised in 1.0.1 (see ``docs/guides/core-vs-ext.md``).  They moved
out of core because HIL is a workflow pattern, not a primitive every
agent needs, and the code surface (~600 LoC combined) is non-trivial.

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
