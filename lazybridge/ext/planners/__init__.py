"""Orchestrator factories — give an LLM sub-agents and a planning toolkit.

This is an **extension** module under the LazyBridge core-vs-ext policy
(see ``docs/guides/core-vs-ext.md``). API may change between minor
releases; consult the per-module CHANGELOG before pinning.

**Naming.** "Orchestrator" intentionally avoids the verbal collision
with :class:`lazybridge.Plan` (the *static* DAG engine).  An
**orchestrator** is an LLM agent that *dynamically* dispatches to
sub-agents — same input shape (``agents: list[Agent]``), different
runtime behaviour from a declared :class:`Plan`.  The previous names
(``make_planner`` / ``make_blackboard_planner``) remain available as
aliases for backward compatibility.

Two factories, same input shape, different trade-offs:

* :func:`orchestrator_agent` (in :mod:`lazybridge.ext.planners.builder`)
  — DAG builder. The LLM composes a :class:`lazybridge.Plan` one step at
  a time via five validated builder tools (``create_plan``, ``add_step``,
  ``inspect_plan``, ``run_plan``, ``discard_plan``). Native parallel.
  Compile-time DAG validation. Optional ``verify=`` judge loop.

* :func:`blackboard_orchestrator_agent` (in
  :mod:`lazybridge.ext.planners.blackboard`) — flat todo list. The LLM
  manages a list of tasks via three blackboard tools (``set_plan``,
  ``get_plan``, ``mark_done``). No DAG, no structural validation.
  Easier to prompt; flexible re-planning.

Pick by trade-off:

- Need parallelism / structural validation / cost-aware verify →
  ``orchestrator_agent``.
- Exploratory work where the shape emerges as you go →
  ``blackboard_orchestrator_agent``.
"""

from lazybridge.ext.planners.blackboard import (
    BLACKBOARD_PLANNER_GUIDANCE,
    blackboard_orchestrator_agent,
    make_blackboard_planner,
)
from lazybridge.ext.planners.builder import (
    PLANNER_GUIDANCE,
    PLANNER_VERIFY_PROMPT,
    PlanSpec,
    StepSpec,
    make_execute_plan_tool,
    make_plan_builder_tools,
    make_planner,
    orchestrator_agent,
)

__all__ = [
    # Canonical names (post unified-surface rename).
    "orchestrator_agent",
    "blackboard_orchestrator_agent",
    # Backward-compat aliases — same callables, the older names.
    "make_planner",
    "make_blackboard_planner",
    # Shared toolkit / prompts.
    "make_plan_builder_tools",
    "make_execute_plan_tool",
    "PLANNER_GUIDANCE",
    "PLANNER_VERIFY_PROMPT",
    "PlanSpec",
    "StepSpec",
    "BLACKBOARD_PLANNER_GUIDANCE",
]
