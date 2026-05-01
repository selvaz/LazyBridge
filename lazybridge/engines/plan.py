"""PlanEngine — structured multi-step execution with compile-time validation."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.sentinels import (
    Sentinel,
    _FromParallel,
    _FromParallelAll,
    _FromPrev,
    _FromStart,
    _FromStep,
    from_prev,
)
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.store import Store
    from lazybridge.tools import Tool


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
# PlanCompiler — build-time validation
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
        pos: dict[str, int] = {s.name: i for i, s in enumerate(steps)}
        # Position-keyed parallel flag for from_parallel_all band-start checks.
        is_parallel: dict[str, bool] = {s.name: bool(s.parallel) for s in steps}

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


# ---------------------------------------------------------------------------
# Plan — the engine
# ---------------------------------------------------------------------------


class Plan:
    """Structured multi-step execution engine.

    Steps run sequentially by default.  Routing is **explicit** at the
    Step level via ``Step(routes={...})`` (predicate map) or
    ``Step(routes_by="field")`` (LLM-decided via a Literal field on
    the structured output).  Parallel branches via ``step.parallel=True``.

    PlanCompiler runs at Agent construction time; errors surface before any LLM call.
    """

    def __init__(
        self,
        *steps: Step,
        max_iterations: int = 100,
        store: Store | None = None,
        checkpoint_key: str | None = None,
        resume: bool = False,
        on_concurrent: Literal["fail", "fork"] = "fail",
    ) -> None:
        """Construct a Plan.

        Checkpoint / resume
        -------------------
        Pass ``store=`` and ``checkpoint_key=`` to persist minimal plan state
        (next step to run, ``writes``-bucket values, completed step names)
        after every step. Pass ``resume=True`` together with a populated
        ``store[checkpoint_key]`` to pick up where the previous run stopped
        (useful after a crash, interrupt, or external pause).

        Example::

            store = Store(db="run.sqlite")
            plan = Plan(
                Step(researcher, writes="research"),
                Step(writer, writes="draft"),
                store=store,
                checkpoint_key="my_pipeline",
                resume=True,
            )
            Agent(engine=plan)("topic")   # continues if a checkpoint exists

        The persisted shape is intentionally small (no Envelopes, no
        in-memory history): ``{"next_step": str, "kv": {...},
        "completed_steps": [...], "status": str, "run_uid": str}``.
        The in-memory ``history`` restarts empty on resume — only
        ``writes``-bucket values survive across process boundaries.

        Concurrency
        -----------
        Every checkpoint write goes through
        :meth:`lazybridge.store.Store.compare_and_swap`, so two Plan
        runs can never silently overwrite each other's state.  Two
        policies are available via ``on_concurrent=``:

        * ``"fail"`` (default) — ``checkpoint_key`` identifies a single
          in-flight run.  A second Plan on the same key, while the first
          is still running, raises :class:`ConcurrentPlanRunError`.
          This is the correctness floor; pick it when runs legitimately
          share state (e.g. graceful crash-resume via ``resume=True``).

        * ``"fork"`` — ``checkpoint_key`` names the *pipeline*; each
          ``.run()`` claims its own isolated effective key
          ``f"{checkpoint_key}:{run_uid}"``.  Many runs of the same
          pipeline can execute concurrently with no collision.  This
          is the mode you want for fan-out workflows (N backtests /
          seeds / tickers sharing a pipeline definition).  ``resume``
          is not supported in ``fork`` mode because there is no single
          shared checkpoint to resume — if you need resume, use
          ``on_concurrent="fail"`` with distinct per-run keys.

        Example::

            store = Store(db="run.sqlite")
            plan = Plan(
                Step(researcher, writes="research"),
                Step(writer, writes="draft"),
                store=store,
                checkpoint_key="my_pipeline",
                resume=True,
            )
        """
        if on_concurrent not in ("fail", "fork"):
            raise ValueError(
                f"Plan(on_concurrent={on_concurrent!r}): must be one of "
                f"'fail' (default, raise on collision) or 'fork' (isolate "
                f"each run under a suffixed key)."
            )
        if on_concurrent == "fork" and resume:
            raise ValueError(
                "Plan(on_concurrent='fork', resume=True) is not supported: "
                "'fork' gives each run its own key, so there is no single "
                "shared checkpoint to resume from.  Use on_concurrent='fail' "
                "with a unique per-run checkpoint_key if you need resume."
            )
        self.steps = list(steps)
        self.max_iterations = max_iterations
        self._compiler = PlanCompiler()
        self.store = store
        self.checkpoint_key = checkpoint_key
        self.resume = resume
        self.on_concurrent = on_concurrent
        # Validation deferred to Agent.__init__ after tools are resolved

    def _validate(self, tool_map: dict[str, Tool]) -> None:
        self._compiler.validate(self.steps, tool_map)

    def _step_map(self) -> dict[str, Step]:
        return {s.name: s for s in self.steps if s.name}

    @staticmethod
    def _aggregate_nested_metadata(
        result_env: Envelope,
        history: list[StepResult],
    ) -> Envelope:
        """Fold every prior step's cost/tokens into ``result_env.metadata.nested_*``.

        The outer envelope's direct metadata (``input_tokens``,
        ``output_tokens``, ``cost_usd``, …) continues to describe
        *this* — the final step's — cost; everything upstream is summed
        into ``nested_input_tokens`` / ``nested_output_tokens`` /
        ``nested_cost_usd`` so a caller reading

            total = env.metadata.input_tokens + env.metadata.output_tokens \
                    + env.metadata.nested_input_tokens \
                    + env.metadata.nested_output_tokens

        sees the full pipeline spend regardless of how many Plan steps
        produced it.  A step's own ``nested_*`` (tokens absorbed by
        agents-as-tools it invoked) roll up with it, so multi-level
        pipelines compose cleanly.

        Identity-skips ``result_env`` if it's already in ``history`` so
        the final step isn't double-counted.
        """
        nested_in = result_env.metadata.nested_input_tokens
        nested_out = result_env.metadata.nested_output_tokens
        nested_cost = result_env.metadata.nested_cost_usd
        added = False
        for sr in history:
            if sr.envelope is result_env:
                continue  # don't double-count the tail
            m = sr.envelope.metadata
            nested_in += m.input_tokens + m.nested_input_tokens
            nested_out += m.output_tokens + m.nested_output_tokens
            nested_cost += m.cost_usd + m.nested_cost_usd
            added = True
        if not added:
            return result_env
        new_meta = result_env.metadata.model_copy(
            update={
                "nested_input_tokens": nested_in,
                "nested_output_tokens": nested_out,
                "nested_cost_usd": nested_cost,
            }
        )
        return result_env.model_copy(update={"metadata": new_meta})

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_store(self) -> Store | None:
        return self.store if self.checkpoint_key else None

    def _effective_key(self, run_uid: str) -> str | None:
        """Return the store key actually used for this run's checkpoint.

        * ``on_concurrent="fail"`` → the user-supplied ``checkpoint_key``
          (single-writer semantics; two concurrent runs collide → CAS
          raises :class:`ConcurrentPlanRunError`).
        * ``on_concurrent="fork"`` → ``f"{checkpoint_key}:{run_uid}"`` so
          every ``.run()`` lives in its own namespace.
        """
        if self.checkpoint_key is None:
            return None
        if self.on_concurrent == "fork":
            return f"{self.checkpoint_key}:{run_uid}"
        return self.checkpoint_key

    def _save_checkpoint(
        self,
        *,
        effective_key: str | None,
        last_snapshot: dict[str, Any] | None,
        next_step: str | None,
        kv: dict[str, Any],
        completed: list[str],
        status: str,
        run_uid: str,
    ) -> dict[str, Any] | None:
        """CAS-aware checkpoint write.  Returns the snapshot written, or
        ``None`` when no checkpoint store is configured.

        Each call executes ``compare_and_swap(checkpoint_key, last_snapshot,
        new_snapshot)`` so two concurrent Plan runs sharing a key
        deterministically converge: the first writer wins, the second
        raises :class:`ConcurrentPlanRunError` instead of silently
        overwriting.  ``last_snapshot`` is the value we previously wrote
        (or read via :meth:`_claim_checkpoint`) — threading it through
        avoids a read-modify-write window.
        """
        store = self._checkpoint_store()
        if store is None or effective_key is None:
            return None
        # Snapshot the mutable buckets at write time.  Without these
        # copies, the returned ``new_snap`` would share its ``kv`` /
        # ``completed_steps`` references with the live mutating values
        # in ``Plan.run``.  The next iteration's mutation (``kv[step
        # .writes] = ...``) would change ``last_snap`` retroactively;
        # the subsequent CAS would compare the mutated last_snap to
        # the previously-written-to-disk JSON and report a false
        # collision (``ConcurrentPlanRunError`` against our own
        # run_uid).  The fix is mechanical — break the aliasing.
        new_snap: dict[str, Any] = {
            "next_step": next_step,
            "kv": dict(kv),
            "completed_steps": list(completed),
            "status": status,
            "run_uid": run_uid,
        }
        if not store.compare_and_swap(effective_key, last_snapshot, new_snap):
            raise ConcurrentPlanRunError(
                f"Checkpoint {effective_key!r} was modified by another "
                f"writer mid-run (our run_uid={run_uid!r}).  Two Plan runs "
                f"appear to share this key — use a unique checkpoint_key "
                f"per concurrent run, or pass on_concurrent='fork'."
            )
        return new_snap

    def _load_checkpoint(self, effective_key: str | None) -> dict[str, Any] | None:
        store = self._checkpoint_store()
        if store is None or effective_key is None or not self.resume:
            return None
        saved = store.read(effective_key)
        if not isinstance(saved, dict):
            return None
        return saved

    def _claim_checkpoint(
        self,
        effective_key: str | None,
        run_uid: str,
    ) -> dict[str, Any] | None:
        """Acquire ownership of ``checkpoint_key`` for this run.

        * Fresh run, key absent or a prior ``status=="done"`` checkpoint
          → CAS-write a ``status="claimed"`` placeholder up-front so
          two concurrent fresh runs collide here, before either has
          executed any step.
        * ``resume=True`` and an in-flight checkpoint exists → adopt it,
          stamping our ``run_uid`` via CAS so subsequent saves compare
          against us rather than the crashed run.
        * In-flight checkpoint and ``resume=False`` → raise
          :class:`ConcurrentPlanRunError`.
        """
        store = self._checkpoint_store()
        if store is None or effective_key is None:
            return None
        existing = store.read(effective_key)
        # Build a "claimed" placeholder snapshot.  next_step / kv /
        # completed_steps are intentionally empty — the first real
        # ``_save_checkpoint`` after the first step will overwrite via
        # CAS that compares against this placeholder.
        first_step = self.steps[0].name if self.steps else None
        claimed_snap = {
            "next_step": first_step,
            "kv": {},
            "completed_steps": [],
            "status": "claimed",
            "run_uid": run_uid,
        }
        if not isinstance(existing, dict):
            # Fresh run — claim via CAS from "key must not exist" (None).
            # A second concurrent fresh run loses this CAS and fails fast.
            if not store.compare_and_swap(effective_key, None, claimed_snap):
                raise ConcurrentPlanRunError(
                    f"Lost race claiming {effective_key!r} — another fresh "
                    f"run wrote the key between our read and claim.  Retry "
                    f"with a unique checkpoint_key, or pass "
                    f"on_concurrent='fork'."
                )
            return claimed_snap
        status = existing.get("status")
        if status == "done":
            # Prior run finished cleanly.  Two sub-cases:
            #  * resume=True → DO NOT claim; return the done snap so the
            #    caller short-circuits to the cached ``kv`` (this is the
            #    documented "resume after done" no-op).
            #  * resume=False → claim by CAS-overwriting the done snap
            #    so concurrent fresh re-runs serialise on the same key.
            if self.resume:
                return existing
            if not store.compare_and_swap(effective_key, existing, claimed_snap):
                raise ConcurrentPlanRunError(
                    f"Lost race claiming completed key {effective_key!r} — "
                    f"another run moved past 'done' before we could claim. "
                    f"Retry."
                )
            return claimed_snap
        if status is None:
            # Key holds a non-plan value (user mis-configured the store).
            # Don't try to CAS over arbitrary data — surface clearly.
            raise ConcurrentPlanRunError(
                f"Checkpoint key {effective_key!r} holds a value with no "
                f"'status' field; refusing to overwrite.  Use a different "
                f"checkpoint_key or a dedicated Store."
            )
        if not self.resume:
            hint = (
                "Pass on_concurrent='fork' to give each run its own key, "
                "or use a unique checkpoint_key per concurrent run."
                if self.on_concurrent == "fail"
                else "This key should not be shared under fork mode — "
                "investigate the code path that produced the collision."
            )
            raise ConcurrentPlanRunError(
                f"Checkpoint {effective_key!r} is already held by "
                f"run_uid={existing.get('run_uid')!r} (status={status!r}).  "
                f"{hint}"
            )
        # Adopt: CAS from the existing state to the same shape with our
        # run_uid stamped in, so concurrent saves compare against us.
        adopted = {**existing, "run_uid": run_uid}
        if not store.compare_and_swap(effective_key, existing, adopted):
            raise ConcurrentPlanRunError(
                f"Lost race claiming {effective_key!r} for resume — "
                f"another run stamped it between our read and claim.  Retry."
            )
        return adopted

    async def run(
        self,
        env: Envelope,
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Store | None = None,
        plan_state: PlanState | None = None,
    ) -> Envelope:
        run_id = str(uuid.uuid4())
        run_uid = uuid.uuid4().hex  # checkpoint ownership stamp (CAS guard)
        tool_map = {t.name: t for t in tools}
        step_map = self._step_map()

        # The effective store key differs by concurrency policy:
        #   fail → checkpoint_key as-is (one in-flight run per key).
        #   fork → f"{checkpoint_key}:{run_uid}" (every run isolated).
        effective_key = self._effective_key(run_uid)

        # Claim the effective key up-front via CAS so two concurrent
        # Plan runs colliding on the same key fail fast instead of
        # corrupting each other.  ``last_snap`` is then threaded through
        # every save so subsequent CAS operations compare against what
        # WE last wrote (or adopted on resume).
        last_snap = self._claim_checkpoint(effective_key, run_uid)

        # Resume: prefer explicit plan_state over store-backed checkpoint
        checkpoint = self._load_checkpoint(effective_key) if plan_state is None else None
        history: list[StepResult] = list(plan_state.history) if plan_state else []
        kv: dict[str, Any] = dict(plan_state.store) if plan_state else {}
        completed: list[str] = []
        if checkpoint is not None:
            kv.update(checkpoint.get("kv") or {})
            completed = list(checkpoint.get("completed_steps") or [])

        # Resume from checkpoint if available
        current_name: str | None
        if plan_state and plan_state.next_step:
            current_name = plan_state.next_step
        elif checkpoint and checkpoint.get("next_step"):
            current_name = checkpoint["next_step"]
        elif checkpoint and checkpoint.get("status") == "done":
            # Already finished — return a stub envelope with kv.
            return Envelope(task=env.task, context=env.context, payload=kv)
        elif self.steps:
            current_name = self.steps[0].name
        else:
            return env

        prev_env = env
        start_env = env
        iterations = 0
        effective_store = store or self.store

        all_step_names = [s.name for s in self.steps]

        while current_name and iterations < self.max_iterations:
            iterations += 1
            step = step_map.get(current_name)
            if not step:
                break

            # ──────────────────────────────────────────────────────────────
            # Parallel branch dispatch
            # ──────────────────────────────────────────────────────────────
            # When the current step is ``parallel=True``, collect every
            # consecutive ``parallel=True`` step in the DECLARED order and
            # run them concurrently via ``asyncio.gather``.  Each branch
            # sees the SAME ``prev_env`` / ``history`` / ``kv`` snapshot —
            # a branch cannot observe its siblings' effects.  State updates
            # apply sequentially after the gather so ``writes`` are
            # deterministic.  Routing (``routes`` / ``routes_by``) is
            # ignored on parallel branches — control flow after the group
            # falls through to the
            # next declared step in linear order (the conventional "join").
            if step.parallel:
                try:
                    idx = all_step_names.index(step.name)
                except ValueError:
                    idx = -1
                group = [step]
                while idx + 1 < len(all_step_names):
                    idx += 1
                    nxt = step_map.get(all_step_names[idx] or "")
                    if nxt and nxt.parallel:
                        group.append(nxt)
                    else:
                        idx -= 1  # don't consume the non-parallel step
                        break

                # Each branch gets an isolated snapshot of history/kv.
                hist_snap = list(history)
                kv_snap = dict(kv)
                raw = await asyncio.gather(
                    *[
                        self._execute_one(
                            s,
                            prev_env,
                            start_env,
                            hist_snap,
                            kv_snap,
                            tool_map=tool_map,
                            session=session,
                            run_id=run_id,
                            branch_id=s.name,
                        )
                        for s in group
                    ],
                    return_exceptions=True,
                )

                # Atomicity: scan ALL branches for failure first.  If any
                # branch errored we return WITHOUT applying any writes to
                # ``kv`` / ``effective_store`` / ``history`` / ``completed``
                # — so a later resume re-runs the whole band cleanly
                # instead of partially-double-applying side-effects from
                # siblings that succeeded earlier in the iteration order.
                first_failure_step: str | None = None
                first_failure_env: Envelope | None = None
                for s, r in zip(group, raw):
                    step_name = s.name or ""
                    if isinstance(r, BaseException):
                        first_failure_step = step_name
                        first_failure_env = Envelope.error_envelope(r)
                        break
                    if r.error is not None:
                        first_failure_step = step_name
                        first_failure_env = r
                        break
                if first_failure_env is not None:
                    last_snap = self._save_checkpoint(
                        effective_key=effective_key,
                        last_snapshot=last_snap,
                        next_step=first_failure_step,
                        kv=kv,
                        completed=completed,
                        status="failed",
                        run_uid=run_uid,
                    )
                    return self._aggregate_nested_metadata(first_failure_env, history)

                # All branches succeeded — apply writes in declared order.
                last_ok: Envelope | None = None
                for s, r in zip(group, raw):
                    step_name = s.name or ""
                    # Type-narrow: failure-scan above already returned on
                    # BaseException / r.error; remaining ``r`` are Envelopes.
                    assert isinstance(r, Envelope)
                    if s.writes and r.payload is not None:
                        kv[s.writes] = r.payload
                        if effective_store:
                            effective_store.write(s.writes, r.payload)
                    history.append(StepResult(step_name=step_name, envelope=r))
                    completed.append(step_name)
                    last_ok = r

                # Advance past the whole parallel group.  Routing
                # primitives (``routes`` / ``routes_by``) are NOT
                # consulted on parallel branches — control flow after
                # a band always falls through linearly.
                current_name = all_step_names[idx + 1] if idx + 1 < len(all_step_names) else None
                prev_env = last_ok or prev_env
                last_snap = self._save_checkpoint(
                    effective_key=effective_key,
                    last_snapshot=last_snap,
                    next_step=current_name,
                    kv=kv,
                    completed=completed,
                    status="running" if current_name else "done",
                    run_uid=run_uid,
                )
                continue

            # ──────────────────────────────────────────────────────────────
            # Sequential path
            # ──────────────────────────────────────────────────────────────
            result_env = await self._execute_one(
                step,
                prev_env,
                start_env,
                history,
                kv,
                tool_map=tool_map,
                session=session,
                run_id=run_id,
            )

            # If the step errored, persist a "failed" checkpoint that
            # points back at the same step so a future resume= retries it.
            if result_env.error is not None:
                last_snap = self._save_checkpoint(
                    effective_key=effective_key,
                    last_snapshot=last_snap,
                    next_step=step.name,
                    kv=kv,
                    completed=completed,
                    status="failed",
                    run_uid=run_uid,
                )
                return self._aggregate_nested_metadata(result_env, history)

            # Persist writes
            if step.writes and result_env.payload is not None:
                kv[step.writes] = result_env.payload
                if effective_store:
                    effective_store.write(step.writes, result_env.payload)

            history.append(StepResult(step_name=step.name or "", envelope=result_env))
            completed.append(step.name or "")
            prev_env = result_env

            # Determine next step via routes / routes_by or linear progression.
            current_name = self._routing(result_env, step, step_map)

            # Save checkpoint after each step so a crash mid-plan can resume
            last_snap = self._save_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snap,
                next_step=current_name,
                kv=kv,
                completed=completed,
                status="running" if current_name else "done",
                run_uid=run_uid,
            )

        # If we exited the loop because the iteration cap was hit (rather
        # than because the plan ran to completion with ``current_name``
        # going None), surface that as an error envelope instead of
        # quietly returning the last partial result — which previously
        # made a routing cycle or runaway plan look successful.
        if current_name and iterations >= self.max_iterations:
            err = ErrorInfo(
                type="MaxIterationsExceeded",
                message=(
                    f"Plan exceeded max_iterations={self.max_iterations} "
                    f"while routing; last step was {current_name!r}.  "
                    f"Suspect a routing cycle (routes / routes_by pointing "
                    f"at an earlier step) or an under-sized max_iterations."
                ),
                retryable=False,
            )
            err_env = Envelope(
                task=prev_env.task,
                context=prev_env.context,
                payload=prev_env.payload,
                metadata=prev_env.metadata,
                error=err,
            )
            return self._aggregate_nested_metadata(err_env, history)

        return self._aggregate_nested_metadata(prev_env, history)

    async def _execute_one(
        self,
        step: Step,
        prev_env: Envelope,
        start_env: Envelope,
        history: list[StepResult],
        kv: dict[str, Any],
        *,
        tool_map: dict[str, Tool],
        session: Session | None,
        run_id: str,
        branch_id: str | None = None,
    ) -> Envelope:
        """Resolve sentinels, build the step env, and execute the step.

        Returns a normalised ``Envelope``.  Does NOT mutate ``history`` /
        ``kv`` — the caller applies those deterministically so parallel
        branches see a consistent snapshot.
        ``branch_id`` is set for parallel-branch steps so their Session
        events can be distinguished from sequential-step events.
        """
        step_task_env = self._resolve_sentinel(step.task, prev_env, start_env, history, kv)

        # Short-circuit when the referenced upstream envelope carries an
        # error.  Previously the resolver stripped ``.error`` and this
        # step ran on the error's ``text()`` as its task — masking the
        # real failure and producing a downstream cascade that was hard
        # to diagnose.  Now we propagate the error envelope verbatim.
        if step_task_env.error is not None:
            return Envelope(
                task=step_task_env.task,
                context=step_task_env.context,
                payload=step_task_env.payload,
                metadata=step_task_env.metadata,
                error=step_task_env.error,
            )

        ctx_parts: list[str] = []
        if step.context is not None:
            # ``context=`` accepts a single Sentinel/str OR a list of them.
            # The list form lets a step pull data from multiple upstream
            # steps without inserting an intermediate combiner — each item
            # resolves independently and the parts are joined with the
            # same blank-line separator we use for ``sources``.  A single
            # item normalises to a 1-list so the resolver path is uniform.
            items = step.context if isinstance(step.context, list) else [step.context]
            for item in items:
                ctx_env = self._resolve_sentinel(item, prev_env, start_env, history, kv)
                if ctx_env.context:
                    ctx_parts.append(ctx_env.context)
                if ctx_env.payload and isinstance(ctx_env.payload, str):
                    ctx_parts.append(ctx_env.payload)
        for src in step.sources:
            if hasattr(src, "text"):
                ctx_parts.append(src.text())

        merged_ctx = "\n\n".join(ctx_parts) if ctx_parts else None
        step_env = Envelope(task=step_task_env.task, context=merged_ctx, payload=step_task_env.payload)

        result_env = await self._exec_step(
            step,
            step_env,
            tool_map=tool_map,
            session=session,
            run_id=run_id,
            branch_id=branch_id,
        )
        return Envelope(
            task=step_env.task,
            context=step_env.context,
            payload=result_env if not isinstance(result_env, Envelope) else result_env.payload,
            metadata=result_env.metadata if isinstance(result_env, Envelope) else EnvelopeMetadata(),
            error=result_env.error if isinstance(result_env, Envelope) else None,
        )

    def _resolve_sentinel(
        self,
        sentinel: Sentinel | str,
        prev: Envelope,
        start: Envelope,
        history: list[StepResult],
        kv: dict[str, Any],
    ) -> Envelope:
        # ``from_prev`` means "the previous step's *output* becomes the next
        # step's task" — i.e. real chain semantics.  Without this promotion
        # the default behaviour was for every step to receive the original
        # user task, so ``Plan(Step(a), Step(b))`` was NOT actually a chain.
        # We carry metadata AND ``.error`` through so downstream code can
        # short-circuit rather than silently feed an errored payload into
        # the next step.
        if isinstance(sentinel, _FromPrev):
            return Envelope(
                task=prev.text(),
                context=prev.context,
                payload=prev.payload,
                metadata=prev.metadata,
                error=prev.error,
            )
        if isinstance(sentinel, _FromStart):
            return start
        # from_step / from_parallel also forward the referenced step's
        # output as the next step's task for consistency with from_prev.
        if isinstance(sentinel, _FromStep):
            for r in reversed(history):
                if r.step_name == sentinel.name:
                    e = r.envelope
                    return Envelope(
                        task=e.text(),
                        context=e.context,
                        payload=e.payload,
                        metadata=e.metadata,
                        error=e.error,
                    )
            return start
        if isinstance(sentinel, _FromParallel):
            for r in reversed(history):
                if r.step_name == sentinel.name:
                    e = r.envelope
                    return Envelope(
                        task=e.text(),
                        context=e.context,
                        payload=e.payload,
                        metadata=e.metadata,
                        error=e.error,
                    )
            return start
        if isinstance(sentinel, _FromParallelAll):
            return self._aggregate_parallel_band(sentinel.name, history, fallback=start)
        if isinstance(sentinel, str):
            return Envelope(task=sentinel, payload=sentinel)
        return prev

    def _aggregate_parallel_band(
        self,
        start_name: str,
        history: list[StepResult],
        *,
        fallback: Envelope,
    ) -> Envelope:
        """Build a single Envelope from every consecutive parallel sibling
        starting at ``start_name`` (in declared order).

        Returns an envelope where ``task`` and ``payload`` are both a
        labelled-text join — ``"[branch_a]\\n<text>\\n\\n[branch_b]\\n<text>..."``
        — so the next step's agent reads all branches via ``env.text()``
        without any change to its tool. The first non-None branch error
        is propagated so downstream can short-circuit.

        Per-branch cost is already accumulated by the engine via
        ``_aggregate_nested_metadata`` from ``history``, so this function
        does not re-sum tokens (would double-count).

        If the named step isn't found in the plan, returns ``fallback``.
        Compile-time validation rejects non-parallel start names, so the
        runtime degenerate case (band of size 1) shouldn't fire — but the
        function tolerates it to keep the engine non-crashy.
        """
        # Find the contiguous parallel band starting at start_name.
        names = [s.name or "" for s in self.steps]
        try:
            start_idx = names.index(start_name)
        except ValueError:
            return fallback

        # Walk forward while consecutive steps remain parallel.
        band_names: list[str] = []
        for i in range(start_idx, len(self.steps)):
            s = self.steps[i]
            if i > start_idx and not s.parallel:
                break
            band_names.append(s.name or "")
            if not s.parallel:
                # Degenerate single-step case (compiler should have rejected).
                break

        # For each band member, pick the most recent matching history entry.
        branch_envs: list[tuple[str, Envelope]] = []
        for n in band_names:
            for r in reversed(history):
                if r.step_name == n:
                    branch_envs.append((n, r.envelope))
                    break

        if not branch_envs:
            return fallback

        # Labelled-text join — both ``task`` (next step's prompt) and
        # ``payload`` (so ``Envelope.text()`` returns it) carry the join.
        sections = [f"[{n}]\n{e.text() if not e.error else f'(error) {e.error.message}'}" for n, e in branch_envs]
        joined = "\n\n".join(sections)

        # Short-circuit: first error wins so downstream can detect failure.
        first_error = next((e.error for _, e in branch_envs if e.error), None)

        return Envelope(
            task=joined,
            context=None,
            payload=joined,
            error=first_error,
        )

    async def _exec_step(
        self,
        step: Step,
        env: Envelope,
        *,
        tool_map: dict[str, Tool],
        session: Session | None,
        run_id: str,
        branch_id: str | None = None,
    ) -> Envelope:
        if session:
            payload: dict[str, Any] = {"step": step.name, "task": env.task}
            if branch_id is not None:
                payload["branch_id"] = branch_id
            session.emit(EventType.TOOL_CALL, payload, run_id=run_id)

        try:
            target = step.target
            if isinstance(target, str):
                tool = tool_map.get(target)
                if tool is None:
                    raise RuntimeError(f"Tool {target!r} not found")
                task_str = env.task or env.text()
                raw = await tool.run(
                    **{"task": task_str}
                    if "task" in tool.definition().parameters.get("properties", {})
                    else _first_arg_kwargs(tool, task_str)
                )
                # Preserve the inner Envelope (agent-as-tool) so metadata
                # survives the step boundary; otherwise wrap the raw value.
                if isinstance(raw, Envelope):
                    result_env = Envelope(
                        task=env.task,
                        context=raw.context or env.context,
                        payload=raw.payload,
                        metadata=raw.metadata,
                        error=raw.error,
                    )
                else:
                    result_env = Envelope(task=env.task, payload=raw)
            elif callable(target) and hasattr(target, "run") and hasattr(target, "_is_lazy_agent"):
                # Agent as step
                result_env = await target.run(env)
            elif callable(target):
                # Raw callable
                import asyncio as _asyncio

                if _asyncio.iscoroutinefunction(target):
                    raw = await target(env.task or env.text())
                else:
                    raw = target(env.task or env.text())
                result_env = Envelope(task=env.task, payload=raw)
            else:
                raise RuntimeError(f"Cannot execute step target: {target!r}")

            if session:
                result_payload: dict[str, Any] = {"step": step.name, "result": result_env.text()[:200]}
                if branch_id is not None:
                    result_payload["branch_id"] = branch_id
                session.emit(EventType.TOOL_RESULT, result_payload, run_id=run_id)
            return result_env

        except Exception as exc:
            if session:
                err_payload: dict[str, Any] = {"step": step.name, "error": str(exc)}
                if branch_id is not None:
                    err_payload["branch_id"] = branch_id
                session.emit(EventType.TOOL_ERROR, err_payload, run_id=run_id)
            return Envelope.error_envelope(exc)

    def _routing(
        self,
        result_env: Envelope,
        step: Step,
        step_map: dict[str, Step],
    ) -> str | None:
        """Determine the next step's name, or ``None`` to end the Plan.

        Routing is **explicit and visible at the Step level**: ``routes=``
        (predicate map) or ``routes_by=`` (field name on the structured
        output) declares the branches.  Falls through to linear
        progression when no branch matches.

        After a routed-to step runs, linear progression resumes from
        its declared position — routing is a *detour*, not a "no
        fall-through" mode.  To make a step terminal, place it last in
        the declared step list.
        """
        # 1. Predicate-based routing.  First matching predicate wins.
        if step.routes:
            for target_name, predicate in step.routes.items():
                try:
                    if predicate(result_env):
                        return target_name
                except Exception as exc:
                    # A misbehaving predicate is a bug, not a runtime
                    # condition — surface it instead of silently
                    # falling through to linear progression and
                    # masking the failure.
                    raise PlanCompileError(
                        f"Step {step.name!r}: routes predicate for {target_name!r} raised {type(exc).__name__}: {exc}"
                    ) from exc

        # 2. Field-driven routing.  Read the named attribute off the
        #    payload; if it's a string matching a step name, jump.
        elif step.routes_by:
            payload = result_env.payload
            if payload is not None and hasattr(payload, step.routes_by):
                value = getattr(payload, step.routes_by)
                if isinstance(value, str) and value in step_map:
                    return value
                # ``None`` and any other non-matching value fall through
                # to linear — the model "decided not to route".

        # 3. Linear progression — next declared step, or end of plan.
        steps_list = list(step_map.keys())
        try:
            idx = steps_list.index(step.name or "")
            if idx + 1 < len(steps_list):
                return steps_list[idx + 1]
        except ValueError:
            pass
        return None

    # ------------------------------------------------------------------
    # Serialisation — to_dict / from_dict  (Plan round-trip)
    # ------------------------------------------------------------------
    #
    # A Plan describes *topology* (steps, routing, writes, parallel flag).
    # Execution targets (functions, agents) live in Python and cannot be
    # serialised directly; ``from_dict`` takes a ``registry={name: target}``
    # so the caller explicitly rebinds names to live objects.  Tool-name
    # targets (``target=str``) survive a round-trip as-is because they
    # are already resolved by ``tool_map`` at run time.

    def to_dict(self) -> dict[str, Any]:
        """Serialise the Plan's topology to a JSON-compatible dict.

        Callables and Agents are serialised by ``name`` only — rebind
        them at load time via :meth:`from_dict`'s ``registry`` kwarg.
        Sentinels, writes, parallel flags, iteration limit, and step
        order are preserved faithfully.
        """
        return {
            "version": 1,
            "max_iterations": self.max_iterations,
            "steps": [_step_to_dict(s) for s in self.steps],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        registry: dict[str, Any] | None = None,
    ) -> Plan:
        """Reconstruct a Plan from a ``to_dict`` payload.

        ``registry`` maps serialised target names back to live callables /
        Agents.  Missing entries for non-tool targets raise
        :class:`KeyError` with the offending name — keeping the failure
        loud rather than producing a silently-broken Plan.

        Example::

            saved = plan.to_dict()                     # store somewhere
            ...
            plan = Plan.from_dict(saved, registry={
                "researcher": researcher_agent,
                "fetch":      fetch_function,
            })
        """
        registry = registry or {}
        steps = [_step_from_dict(s, registry) for s in data.get("steps", [])]
        return cls(*steps, max_iterations=data.get("max_iterations", 100))

    # Engine Protocol compatibility
    async def stream(
        self, env: Envelope, *, tools: list, output_type: type, memory: Any, session: Any
    ) -> AsyncIterator[str]:
        result = await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)
        yield result.text()


def _first_arg_kwargs(tool: Tool, value: str) -> dict[str, str]:
    """Build kwargs dict using the first parameter name of the tool."""
    params = tool.definition().parameters.get("properties", {})
    if params:
        first = next(iter(params))
        return {first: value}
    return {"input": value}


# ---------------------------------------------------------------------------
# Serialisation helpers (module-level so users can import them to write
# their own to_yaml / mermaid renderers on top of the topology shape).
# ---------------------------------------------------------------------------


def _target_to_ref(target: Any) -> dict[str, str]:
    """Serialise a Step.target to a ``{"kind": ..., "name": ...}`` ref.

    Tools referenced by name round-trip as-is; callables and Agents are
    recorded by their ``name`` attribute / ``__name__`` so a registry can
    rebind them on load.
    """
    if isinstance(target, str):
        return {"kind": "tool", "name": target}
    if hasattr(target, "_is_lazy_agent"):
        return {"kind": "agent", "name": getattr(target, "name", "agent")}
    if callable(target):
        return {"kind": "callable", "name": getattr(target, "__name__", "anon")}
    return {"kind": "unknown", "name": str(target)}


def _target_from_ref(ref: dict[str, str], registry: dict[str, Any]) -> Any:
    kind = ref.get("kind")
    name = ref.get("name", "")
    if kind == "tool":
        return name  # keep as string — tool_map resolves it at run time
    if name in registry:
        return registry[name]
    raise KeyError(
        f"Plan.from_dict: no entry in registry for {kind} target {name!r}. "
        f"Pass registry={{'{name}': <callable>}} to rebind."
    )


def _sentinel_to_ref(sentinel: Any) -> dict[str, Any] | None:
    if sentinel is None:
        return None
    if isinstance(sentinel, _FromPrev):
        return {"kind": "from_prev"}
    if isinstance(sentinel, _FromStart):
        return {"kind": "from_start"}
    if isinstance(sentinel, _FromStep):
        return {"kind": "from_step", "name": sentinel.name}
    if isinstance(sentinel, _FromParallel):
        return {"kind": "from_parallel", "name": sentinel.name}
    if isinstance(sentinel, _FromParallelAll):
        return {"kind": "from_parallel_all", "name": sentinel.name}
    if isinstance(sentinel, str):
        return {"kind": "literal", "value": sentinel}
    return None


def _sentinel_from_ref(ref: dict[str, Any] | None) -> Sentinel | str:
    if ref is None:
        return from_prev
    from lazybridge.sentinels import from_parallel, from_parallel_all, from_start, from_step

    kind = ref.get("kind")
    if kind == "from_prev":
        return from_prev
    if kind == "from_start":
        return from_start
    if kind == "from_step":
        return from_step(ref["name"])
    if kind == "from_parallel":
        return from_parallel(ref["name"])
    if kind == "from_parallel_all":
        return from_parallel_all(ref["name"])
    if kind == "literal":
        return ref["value"]
    return from_prev


def _step_to_dict(step: Step) -> dict[str, Any]:
    d: dict[str, Any] = {
        "name": step.name,
        "target": _target_to_ref(step.target),
        "task": _sentinel_to_ref(step.task),
        "parallel": step.parallel,
    }
    if step.context is not None:
        # ``context=`` is single-or-list — preserve the shape on disk so
        # ``from_dict`` round-trips faithfully.  A single sentinel/str
        # serialises to one ref dict; a list serialises to a list of
        # ref dicts.
        if isinstance(step.context, list):
            d["context"] = [_sentinel_to_ref(item) for item in step.context]
        else:
            d["context"] = _sentinel_to_ref(step.context)
    if step.writes:
        d["writes"] = step.writes
    if step.routes is not None:
        # Predicates can't be JSON-serialised — record only target step
        # names; ``from_dict`` rebinds via ``registry["routes:<step>:<target>"]``.
        d["routes"] = sorted(step.routes.keys())
    if step.routes_by is not None:
        d["routes_by"] = step.routes_by
    return d


def _step_from_dict(data: dict[str, Any], registry: dict[str, Any]) -> Step:
    target = _target_from_ref(data["target"], registry)
    task = _sentinel_from_ref(data.get("task"))
    context: Sentinel | str | list[Sentinel | str] | None
    if "context" not in data:
        context = None
    else:
        raw = data["context"]
        if isinstance(raw, list):
            context = [_sentinel_from_ref(item) for item in raw]
        else:
            context = _sentinel_from_ref(raw)
    routes: dict[str, Callable[[Any], bool]] | None = None
    if "routes" in data:
        # Predicates live in Python; rebind by registry key
        # ``f"routes:{step_name}:{target}"``.  Missing keys raise
        # ``KeyError`` so the load fails loud.
        step_name = data.get("name", "<unnamed>")
        routes = {}
        for target_name in data["routes"]:
            key = f"routes:{step_name}:{target_name}"
            if key not in registry:
                raise KeyError(
                    f"Plan.from_dict: no entry in registry for "
                    f"{key!r} (predicate for routes={{{target_name!r}: ...}}). "
                    f"Pass registry={{{key!r}: predicate}} to rebind."
                )
            routes[target_name] = registry[key]
    return Step(
        target=target,
        task=task,
        context=context,
        writes=data.get("writes"),
        parallel=data.get("parallel", False),
        name=data.get("name"),
        routes=routes,
        routes_by=data.get("routes_by"),
    )
