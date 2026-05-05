"""Plan dataclasses + structural exception types.

Carved out of the old monolithic ``plan.py`` (W3.1).  Other plan
submodules import from here; nothing in this module imports from
``_compiler`` / ``_serialisation`` / ``_plan`` so the dependency graph
stays one-way (``_types`` → others, never the reverse).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from lazybridge.envelope import Envelope
from lazybridge.sentinels import Sentinel, from_prev

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Step descriptor
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single node in a Plan.

    Args:
        target:  Tool name (str), callable, or Agent. Required.
        task:    Sentinel or str for the step's task. Default: from_prev.
        context: Sentinel, str, or **list of either** for extra context.
                 A list joins its resolved parts with blank-line separators
                 (same shape as ``sources``) so a step can pull data from
                 multiple upstream steps without an intermediate combiner.
                 Each list item is validated independently at compile time.
                 Default: none.
        sources: Live-view objects with a .text() method injected into context.
        writes:  Key under which Envelope.payload is saved in the Store.
        input:   Expected input payload type (PlanCompiler validates).
        output:  Expected output payload type (triggers structured output).
        parallel: True if this step runs concurrently with siblings.
        name:    Override for display / from_step() lookups.

        routes:  **Predicate-based routing**.  Mapping ``{step_name:
                 predicate(envelope) -> bool}``.  After this step runs,
                 predicates are evaluated in declared order; the first
                 one that returns truthy makes the Plan jump to the
                 corresponding step.  If none match (or ``routes`` is
                 ``None``), execution falls through linearly to the
                 next declared step.  Mutually exclusive with
                 ``routes_by``.
        routes_by: **LLM-decided routing via a named field on the
                 step's structured output**.  Pass the attribute name
                 (e.g. ``"kind"``) — Plan reads ``env.payload.<name>``
                 and, if it's a string matching an existing step name,
                 jumps there.  The output model must declare that
                 field as ``Literal["a", "b", ...]`` (or
                 ``Literal[...] | None``); compile-time validation
                 rejects values that don't match a step name.
                 Mutually exclusive with ``routes``.

    Routing is a **detour**.  After the routed-to step runs, linear
    progression resumes from its position in the declared order — no
    "no fall-through after routing" trap.  To make a step terminal,
    place it at the end of the declared step list (linear progression
    past the last step ends the Plan).  Loops are simply routes back
    to an earlier step; ``Plan(max_iterations=...)`` is the safety
    net.
    """

    target: Any
    task: Sentinel | str = field(default_factory=lambda: from_prev)
    context: Sentinel | str | list[Sentinel | str] | None = None
    sources: list[Any] = field(default_factory=list)
    writes: str | None = None
    input: type = Any
    output: type = str
    parallel: bool = False
    name: str | None = None
    # Routing — exactly one (or neither) of these may be set.  See
    # the Step docstring for semantics.
    routes: dict[str, Callable[[Any], bool]] | None = None
    routes_by: str | None = None

    def __post_init__(self) -> None:
        if self.name is None:
            if isinstance(self.target, str):
                self.name = self.target
            elif callable(self.target) and hasattr(self.target, "__name__"):
                self.name = self.target.__name__
            elif hasattr(self.target, "name"):
                self.name = self.target.name
            else:
                self.name = str(id(self.target))


# ---------------------------------------------------------------------------
# PlanState — checkpoint / resume
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    step_name: str
    envelope: Envelope
    ts: float = field(default_factory=time.time)


@dataclass
class PlanState:
    plan_id: str
    current_step: str
    next_step: str | None
    store: dict[str, Any]
    history: list[StepResult]
    status: Literal["running", "paused", "done", "failed"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConcurrentPlanRunError(RuntimeError):
    """Raised when two Plan runs race for the same ``checkpoint_key``.

    Checkpoints are serialised through :meth:`lazybridge.store.Store.compare_and_swap`
    so the first writer wins and any second writer fails fast instead of
    silently overwriting the first run's state.  Derive a unique
    ``checkpoint_key`` per run (e.g. ``f"pipeline-{uuid.uuid4().hex}"``)
    when you need concurrent execution on the same :class:`Store`.
    """


class PlanCompileError(Exception):
    pass
