"""PlanCompiler — build-time DAG validation.

Owns every check that surfaces as :class:`PlanCompileError` at
construction time:

* Duplicate step names.
* Tool existence (``Step(target="some_tool")`` requires the named
  tool in the tool map).
* Sentinel forward-references (``from_step`` / ``from_parallel`` /
  ``from_parallel_all`` pointing at a step that hasn't been declared
  yet).
* ``from_parallel_all`` band-start invariants (target step is
  ``parallel=True`` and is the FIRST member of the band).
* Type compatibility between consecutive steps' ``output`` / ``input``.
* Routing well-formedness (``routes={...}`` keys are step names,
  values are callable; ``routes_by="field"`` references an existing
  ``Literal[str]`` field on the step's output model that contains
  only valid step names).

Carved out of the old monolithic ``plan.py`` (W3.1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, get_args, get_origin, get_type_hints

from lazybridge.engines.plan._types import PlanCompileError, Step
from lazybridge.sentinels import (
    Sentinel,
    _FromParallel,
    _FromParallelAll,
    _FromPrev,
    _FromStart,
    _FromStep,
)

if TYPE_CHECKING:
    from lazybridge.tools import Tool


def _extract_literal_string_values(annotation: Any) -> list[str]:
    """Return the list of string literals from a ``Literal[...]`` or
    ``Literal[...] | None`` / ``Optional[Literal[...]]`` annotation.

    Returns an empty list when the annotation isn't a Literal of
    strings (so the caller can flag a malformed ``routes_by`` field).
    """
    # Direct Literal["a", "b"]
    args = get_args(annotation)
    if not args:
        return []
    # Pure Literal[...] — args are the values (or types).
    origin = get_origin(annotation)
    if origin is Literal:
        return [a for a in args if isinstance(a, str)]
    # Union: walk every arm and recurse; collect string literals from
    # the Literal arm(s).  Handles ``Literal["a"] | None`` and the
    # equivalent ``Optional[Literal["a"]]``.
    found: list[str] = []
    for arm in args:
        found.extend(_extract_literal_string_values(arm))
    return found


class PlanCompiler:
    """Validates a list of Steps at Plan construction time."""

    def validate(self, steps: list[Step], tool_map: dict[str, Tool]) -> None:
        # Duplicate step names — ``_step_map()`` would silently keep the
        # last definition, hiding the first step's edges.  Surface this
        # at compile time so the user can pick distinct names before any
        # LLM call runs.
        seen: set[str] = set()
        duplicates: list[str] = []
        for s in steps:
            # Step.__post_init__ guarantees s.name is not None.
            assert s.name is not None
            if s.name in seen:
                duplicates.append(s.name)
            seen.add(s.name)
        if duplicates:
            raise PlanCompileError(
                f"Plan has duplicate step name(s): {sorted(set(duplicates))}.  "
                f"Step names must be unique — rename collisions or omit "
                f"one of the duplicates."
            )

        # Build a position index so we can reject forward (future)
        # ``from_step`` references — the runtime falls back to the
        # initial envelope when no history exists for the named step,
        # which silently masks misordered plans.
        # Step.__post_init__ guarantees s.name is not None — the assertion
        # above already established it for the iteration above; redo it
        # for these comprehensions so mypy can narrow.
        pos: dict[str, int] = {cast("str", s.name): i for i, s in enumerate(steps)}
        # Position-keyed parallel flag for from_parallel_all band-start checks.
        is_parallel: dict[str, bool] = {cast("str", s.name): bool(s.parallel) for s in steps}

        for i, step in enumerate(steps):
            # Tool exists
            if isinstance(step.target, str) and step.target not in tool_map:
                raise PlanCompileError(
                    f"Step {step.name!r}: tool {step.target!r} not found in tools. Available: {sorted(tool_map)}"
                )

            # ``context=`` accepts a single sentinel/str OR a list of them.
            # Normalise to a list so every check below iterates uniformly;
            # ``None`` becomes an empty list.
            context_items: list[Sentinel | str]
            if step.context is None:
                context_items = []
            elif isinstance(step.context, list):
                context_items = list(step.context)
            else:
                context_items = [step.context]

            # Each item in a context list must be a known sentinel or a
            # plain string.  Anything else falls through ``_resolve_sentinel``
            # to the ``prev`` envelope at runtime — a silent degradation we
            # want to catch at construction.
            _SENTINEL_TYPES = (_FromPrev, _FromStart, _FromStep, _FromParallel, _FromParallelAll)
            for n, item in enumerate(context_items):
                if not isinstance(item, (str, *_SENTINEL_TYPES)):
                    raise PlanCompileError(
                        f"Step {step.name!r}: context[{n}] has type "
                        f"{type(item).__name__!r} — must be a Sentinel "
                        f"(from_prev / from_start / from_step(...) / "
                        f"from_parallel(...) / from_parallel_all(...)) "
                        f"or a literal str."
                    )

            # from_step references valid step …
            if isinstance(step.task, _FromStep) and step.task.name not in pos:
                raise PlanCompileError(
                    f"Step {step.name!r}: task=from_step({step.task.name!r}) references unknown step."
                )
            for n, ctx_item in enumerate(context_items):
                if isinstance(ctx_item, _FromStep) and ctx_item.name not in pos:
                    raise PlanCompileError(
                        f"Step {step.name!r}: context[{n}]=from_step({ctx_item.name!r}) references unknown step."
                    )
            # … and that step must come *before* this one.  A ``from_step``
            # to a future step quietly degrades to the start envelope at
            # runtime, which looks like success but isn't.
            if isinstance(step.task, _FromStep) and pos.get(step.task.name, -1) >= i:
                raise PlanCompileError(
                    f"Step {step.name!r}: task=from_step({step.task.name!r}) "
                    f"references a step that is not earlier in the plan.  "
                    f"from_step targets must be defined before they're used."
                )
            for n, ctx_item in enumerate(context_items):
                if isinstance(ctx_item, _FromStep) and pos.get(ctx_item.name, -1) >= i:
                    raise PlanCompileError(
                        f"Step {step.name!r}: context[{n}]=from_step({ctx_item.name!r}) "
                        f"references a step that is not earlier in the plan.  "
                        f"from_step targets must be defined before they're used."
                    )
            # from_parallel_all: same forward-ref guard plus the band-start
            # check (the named step must itself be parallel=True; otherwise
            # the "band" is one step and from_step would be the right tool).
            sentinels_to_check: list[tuple[str, Any]] = [("task", step.task)]
            for n, ctx_item in enumerate(context_items):
                slot_label = f"context[{n}]" if isinstance(step.context, list) else "context"
                sentinels_to_check.append((slot_label, ctx_item))
            for slot, sentinel in sentinels_to_check:
                if isinstance(sentinel, _FromParallelAll):
                    if sentinel.name not in pos:
                        raise PlanCompileError(
                            f"Step {step.name!r}: {slot}=from_parallel_all({sentinel.name!r}) references unknown step."
                        )
                    if pos[sentinel.name] >= i:
                        raise PlanCompileError(
                            f"Step {step.name!r}: {slot}=from_parallel_all"
                            f"({sentinel.name!r}) references a step that is "
                            f"not earlier in the plan."
                        )
                    if not is_parallel.get(sentinel.name, False):
                        raise PlanCompileError(
                            f"Step {step.name!r}: {slot}=from_parallel_all"
                            f"({sentinel.name!r}) references a non-parallel "
                            f"step.  from_parallel_all aggregates a contiguous "
                            f"parallel band; its target must be the FIRST "
                            f"member of that band (i.e. parallel=True). "
                            f"Use from_step / from_parallel for single-branch reads."
                        )
                    # The target must also be the *first* member of its
                    # parallel band — i.e. either the first step overall or
                    # immediately preceded by a non-parallel step.  Otherwise
                    # the runtime walks forward from a mid-band position and
                    # silently misses the earlier siblings.
                    target_idx = pos[sentinel.name]
                    if target_idx > 0 and steps[target_idx - 1].parallel:
                        raise PlanCompileError(
                            f"Step {step.name!r}: {slot}=from_parallel_all"
                            f"({sentinel.name!r}) must reference the FIRST "
                            f"member of a parallel band, but the step "
                            f"immediately before it ({steps[target_idx - 1].name!r}) "
                            f"is also parallel=True.  Point the sentinel at the "
                            f"earliest parallel step in the band instead."
                        )
            # Type compatibility: previous step output must match this step input
            if i > 0 and step.input is not Any:
                prev = steps[i - 1]
                if prev.output is not str and prev.output is not Any and prev.output != step.input:
                    # Allow Union types
                    origin = get_origin(step.input)
                    if origin is not None:  # Union, list, etc.
                        pass
                    else:
                        raise PlanCompileError(
                            f"Step {step.name!r}: input={step.input.__name__!r} but previous step "
                            f"{prev.name!r} produces output={prev.output if isinstance(prev.output, str) else prev.output.__name__!r}."
                        )
            # ── Routing validation ──────────────────────────────────────
            # routes= and routes_by= are mutually exclusive.
            if step.routes is not None and step.routes_by is not None:
                raise PlanCompileError(
                    f"Step {step.name!r}: routes= and routes_by= are mutually "
                    f"exclusive.  Use predicate-based routing (routes={{...}}) "
                    f"OR field-driven routing (routes_by='attr'), not both."
                )

            # routes={"step_name": predicate} — every key must be a step
            # name; every value must be callable.  Self-loops and
            # backward routes are allowed (for self-correction loops);
            # the only structural requirement is that the target exists.
            if step.routes is not None:
                if not isinstance(step.routes, dict):
                    raise PlanCompileError(
                        f"Step {step.name!r}: routes= must be a dict, got {type(step.routes).__name__}."
                    )
                for target_name, predicate in step.routes.items():
                    if not isinstance(target_name, str):
                        raise PlanCompileError(
                            f"Step {step.name!r}: routes= keys must be step "
                            f"names (str), got {type(target_name).__name__}."
                        )
                    if target_name not in pos:
                        raise PlanCompileError(
                            f"Step {step.name!r}: routes={{{target_name!r}: ...}} "
                            f"references unknown step.  Known steps: "
                            f"{sorted(pos)}."
                        )
                    if not callable(predicate):
                        raise PlanCompileError(
                            f"Step {step.name!r}: routes[{target_name!r}] is "
                            f"not callable; expected a function "
                            f"(envelope) -> bool."
                        )

            # routes_by="field" — the step's output model must declare
            # ``field`` as Literal[...] (or Literal[...] | None) of step
            # names.  Validates target names at compile time.
            if step.routes_by is not None:
                if not isinstance(step.routes_by, str) or not step.routes_by:
                    raise PlanCompileError(
                        f"Step {step.name!r}: routes_by= must be a non-empty string naming a field on the output model."
                    )
                if step.output is str or not isinstance(step.output, type):
                    raise PlanCompileError(
                        f"Step {step.name!r}: routes_by={step.routes_by!r} "
                        f"requires a Pydantic model as output= (got "
                        f"{step.output!r})."
                    )
                hints = get_type_hints(step.output) if hasattr(step.output, "__annotations__") else {}
                if step.routes_by not in hints:
                    raise PlanCompileError(
                        f"Step {step.name!r}: routes_by={step.routes_by!r} "
                        f"but {step.output.__name__!r} has no field of "
                        f"that name.  Declared fields: {sorted(hints)}."
                    )
                # Walk the type to find the Literal arms.  Accept
                # ``Literal[...]`` and ``Optional[Literal[...]]`` /
                # ``Literal[...] | None``.
                literal_values = _extract_literal_string_values(hints[step.routes_by])
                if not literal_values:
                    raise PlanCompileError(
                        f"Step {step.name!r}: routes_by={step.routes_by!r} "
                        f"requires the field to be typed "
                        f"``Literal['a', 'b', ...]`` (optionally union'd "
                        f"with None).  Got annotation "
                        f"{hints[step.routes_by]!r}."
                    )
                for value in literal_values:
                    if value not in pos:
                        raise PlanCompileError(
                            f"Step {step.name!r}: routes_by={step.routes_by!r} "
                            f"includes Literal value {value!r} which is not "
                            f"a known step name.  Known steps: "
                            f"{sorted(pos)}."
                        )

            # after_branches= — exclusive-branch rejoin point.
            # Requires routes= or routes_by= to be set (otherwise there
            # is no branching and the field is meaningless).  The target
            # step must exist and must come AFTER this step so that skipping
            # intermediate steps is unambiguous.
            if step.after_branches is not None:
                if step.routes is None and step.routes_by is None:
                    raise PlanCompileError(
                        f"Step {step.name!r}: after_branches={step.after_branches!r} "
                        f"requires routes= or routes_by= to also be set — "
                        f"after_branches only applies to exclusive routing."
                    )
                if step.after_branches not in pos:
                    raise PlanCompileError(
                        f"Step {step.name!r}: after_branches={step.after_branches!r} "
                        f"references unknown step.  Known steps: {sorted(pos)}."
                    )
                if pos[step.after_branches] <= i:
                    raise PlanCompileError(
                        f"Step {step.name!r}: after_branches={step.after_branches!r} "
                        f"must come after the routing step in the declared order "
                        f"(step {step.name!r} is at position {i}, "
                        f"{step.after_branches!r} is at position "
                        f"{pos[step.after_branches]})."
                    )
