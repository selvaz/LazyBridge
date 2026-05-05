"""``lazybridge.engines.plan`` — structured multi-step execution.

This package replaces the pre-W3.1 single-file ``plan.py`` (1623 LOC).
The public surface is **unchanged**: every name that used to be
``from lazybridge.engines.plan import X`` still resolves the same way,
including the historically-private serialisation helpers
(``_sentinel_to_ref`` / ``_sentinel_from_ref``) that test modules
import directly.

Layout
------
:mod:`._types`
    ``Step`` / ``StepResult`` / ``PlanState`` dataclasses;
    ``ConcurrentPlanRunError`` / ``PlanCompileError`` exceptions.
:mod:`._compiler`
    ``PlanCompiler`` — every check that surfaces as
    ``PlanCompileError`` at construction (duplicate names, missing
    tools, sentinel forward-refs, ``from_parallel_all`` band-start
    invariants, type compatibility, routing well-formedness).
:mod:`._serialisation`
    Module-level ``_target_to_ref`` / ``_target_from_ref`` /
    ``_sentinel_to_ref`` / ``_sentinel_from_ref`` /
    ``_step_to_dict`` / ``_step_from_dict`` / ``_first_arg_kwargs``
    helpers used by ``Plan.to_dict`` / ``Plan.from_dict`` and by
    third-party YAML / Mermaid renderers built on top of the topology
    shape.
:mod:`._plan`
    The ``Plan`` class — async ``run``, parallel-band scheduling,
    sentinel resolution, routing, checkpoint write/load/claim,
    history aggregation, ``run_many`` / ``arun_many`` fan-out, and
    ``to_dict`` / ``from_dict`` instance methods.
"""

from __future__ import annotations

# Public surface — preserved verbatim.
from lazybridge.engines.plan._compiler import (
    PlanCompiler,
    _extract_literal_string_values,
)
from lazybridge.engines.plan._plan import Plan
from lazybridge.engines.plan._serialisation import (
    _first_arg_kwargs,
    _sentinel_from_ref,
    _sentinel_to_ref,
    _step_from_dict,
    _step_to_dict,
    _target_from_ref,
    _target_to_ref,
)
from lazybridge.engines.plan._types import (
    ConcurrentPlanRunError,
    PlanCompileError,
    PlanState,
    Step,
    StepResult,
)

__all__ = [
    # Primary public API.
    "Plan",
    "Step",
    "StepResult",
    "PlanState",
    "PlanCompiler",
    "PlanCompileError",
    "ConcurrentPlanRunError",
    # Historically-private serialisation helpers exposed for test suites
    # and third-party renderers (YAML, Mermaid, etc.).  Kept on the public
    # package surface to preserve backward compatibility with
    # pre-W3.1 import paths.
    "_target_to_ref",
    "_target_from_ref",
    "_sentinel_to_ref",
    "_sentinel_from_ref",
    "_step_to_dict",
    "_step_from_dict",
    "_first_arg_kwargs",
    "_extract_literal_string_values",
]
