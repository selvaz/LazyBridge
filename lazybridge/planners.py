"""Deprecated alias. Import from ``lazybridge.ext.planners`` instead.

The planner factories were moved to :mod:`lazybridge.ext.planners` so they
follow the framework's core-vs-ext policy
(see ``docs/guides/core-vs-ext.md``). This module re-exports everything
from the new location and emits a :class:`DeprecationWarning` on import.

Will be removed in lazybridge 1.2.
"""

import warnings

warnings.warn(
    "lazybridge.planners has moved to lazybridge.ext.planners; "
    "update your import. The shim will be removed in lazybridge 1.2.",
    DeprecationWarning,
    stacklevel=2,
)

from lazybridge.ext.planners import (  # noqa: E402,F401  (re-export under deprecated name)
    BLACKBOARD_PLANNER_GUIDANCE,
    PLANNER_GUIDANCE,
    PLANNER_VERIFY_PROMPT,
    PlanSpec,
    StepSpec,
    make_blackboard_planner,
    make_execute_plan_tool,
    make_plan_builder_tools,
    make_planner,
)

__all__ = [
    "make_planner",
    "make_plan_builder_tools",
    "make_execute_plan_tool",
    "PLANNER_GUIDANCE",
    "PLANNER_VERIFY_PROMPT",
    "PlanSpec",
    "StepSpec",
    "make_blackboard_planner",
    "BLACKBOARD_PLANNER_GUIDANCE",
]
