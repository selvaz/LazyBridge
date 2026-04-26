"""Planner factories — give an LLM sub-agents and a planning toolkit.

This is an **extension** module under the LazyBridge core-vs-ext policy
(see ``docs/guides/core-vs-ext.md``). API may change between minor
releases; consult the per-module CHANGELOG before pinning.

Two factories, same input shape (``agents: list[Agent]``), different
trade-offs:

* :func:`make_planner` (in :mod:`lazybridge.ext.planners.builder`) — DAG
  builder. The LLM composes a :class:`lazybridge.Plan` one step at a
  time via five validated builder tools (``create_plan``, ``add_step``,
  ``inspect_plan``, ``run_plan``, ``discard_plan``). Native parallel.
  Compile-time DAG validation. Optional ``verify=`` judge loop.

* :func:`make_blackboard_planner` (in :mod:`lazybridge.ext.planners.blackboard`) —
  flat todo list. The LLM manages a list of tasks via three blackboard
  tools (``set_plan``, ``get_plan``, ``mark_done``). No DAG, no
  structural validation. Easier to prompt; flexible re-planning.

Pick by trade-off:

- Need parallelism / structural validation / cost-aware verify → ``make_planner``.
- Exploratory work where the shape emerges as you go → ``make_blackboard_planner``.
"""

#: Stability tag — see ``docs/guides/core-vs-ext.md``.
__stability__ = "alpha"
__lazybridge_min__ = "1.0.0"

from lazybridge.ext.planners.blackboard import (
    BLACKBOARD_PLANNER_GUIDANCE,
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
)

__all__ = [
    # DAG builder
    "make_planner",
    "make_plan_builder_tools",
    "make_execute_plan_tool",
    "PLANNER_GUIDANCE",
    "PLANNER_VERIFY_PROMPT",
    "PlanSpec",
    "StepSpec",
    # Blackboard
    "make_blackboard_planner",
    "BLACKBOARD_PLANNER_GUIDANCE",
]
