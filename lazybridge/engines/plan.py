"""PlanEngine — structured multi-step execution with compile-time validation."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin, get_type_hints

from lazybridge.envelope import Envelope, EnvelopeMetadata
from lazybridge.sentinels import (
    Sentinel,
    _FromParallel,
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
        context: Sentinel or str for extra context. Default: none.
        sources: Live-view objects with a .text() method injected into context.
        writes:  Key under which Envelope.payload is saved in the Store.
        input:   Expected input payload type (PlanCompiler validates).
        output:  Expected output payload type (triggers structured output).
        parallel: True if this step runs concurrently with siblings.
        name:    Override for display / from_step() lookups.
    """

    target: Any
    task: Sentinel | str = field(default_factory=lambda: from_prev)
    context: Sentinel | str | None = None
    sources: list[Any] = field(default_factory=list)
    writes: str | None = None
    input: type = Any
    output: type = str
    parallel: bool = False
    name: str | None = None

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


class PlanCompiler:
    """Validates a list of Steps at Plan construction time."""

    def validate(self, steps: list[Step], tool_map: dict[str, Tool]) -> None:
        names = {s.name for s in steps}
        for i, step in enumerate(steps):
            # Tool exists
            if isinstance(step.target, str) and step.target not in tool_map:
                raise PlanCompileError(
                    f"Step {step.name!r}: tool {step.target!r} not found in tools. "
                    f"Available: {sorted(tool_map)}"
                )
            # from_step references valid step
            if isinstance(step.task, _FromStep) and step.task.name not in names:
                raise PlanCompileError(
                    f"Step {step.name!r}: task=from_step({step.task.name!r}) references unknown step."
                )
            if isinstance(step.context, _FromStep) and step.context.name not in names:
                raise PlanCompileError(
                    f"Step {step.name!r}: context=from_step({step.context.name!r}) references unknown step."
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
            # out.next fields must reference existing steps
            if step.output is not str and isinstance(step.output, type):
                hints = get_type_hints(step.output) if hasattr(step.output, "__annotations__") else {}
                next_hint = hints.get("next")
                if next_hint is not None:
                    literal_args = get_args(next_hint)
                    for arg in literal_args:
                        if isinstance(arg, str) and arg not in names:
                            raise PlanCompileError(
                                f"Step {step.name!r}: output.next Literal contains {arg!r} "
                                f"which is not a known step name."
                            )


# ---------------------------------------------------------------------------
# Plan — the engine
# ---------------------------------------------------------------------------


class Plan:
    """Structured multi-step execution engine.

    Steps run sequentially by default. Routing via ``output.next: Literal[...]``
    field in Pydantic output models. Parallel steps via step.parallel=True.

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
        new_snap: dict[str, Any] = {
            "next_step": next_step,
            "kv": kv,
            "completed_steps": completed,
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
        self, effective_key: str | None, run_uid: str,
    ) -> dict[str, Any] | None:
        """Acquire ownership of ``checkpoint_key`` for this run.

        * Fresh run, key absent or a prior ``status=="done"`` checkpoint
          → returns ``None``; the first subsequent ``_save_checkpoint``
          writes the initial state.
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
        if not isinstance(existing, dict):
            return None
        status = existing.get("status")
        if status in (None, "done"):
            # Prior run finished cleanly (or key holds a non-plan value) —
            # safe to overwrite starting from the first save.
            return existing if status == "done" else None
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
        was_routed = False  # True when the current step was reached via out.next, not linear order
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
            # deterministic.  Routing (``out.next``) is ignored on parallel
            # branches — control flow after the group falls through to the
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
                    *[self._execute_one(
                        s, prev_env, start_env, hist_snap, kv_snap,
                        tool_map=tool_map, session=session, run_id=run_id,
                        branch_id=s.name,
                    ) for s in group],
                    return_exceptions=True,
                )

                # Apply writes / history sequentially; bail on first error.
                last_ok: Envelope | None = None
                for s, r in zip(group, raw):
                    step_name = s.name or ""
                    if isinstance(r, BaseException):
                        err_env = Envelope.error_envelope(r)
                        last_snap = self._save_checkpoint(
                            effective_key=effective_key,
                            last_snapshot=last_snap, next_step=step_name,
                            kv=kv, completed=completed, status="failed",
                            run_uid=run_uid,
                        )
                        return err_env
                    if r.error is not None:
                        last_snap = self._save_checkpoint(
                            effective_key=effective_key,
                            last_snapshot=last_snap, next_step=step_name,
                            kv=kv, completed=completed, status="failed",
                            run_uid=run_uid,
                        )
                        return r
                    if s.writes and r.payload is not None:
                        kv[s.writes] = r.payload
                        if effective_store:
                            effective_store.write(s.writes, r.payload)
                    history.append(StepResult(step_name=step_name, envelope=r))
                    completed.append(step_name)
                    last_ok = r

                # Advance past the whole parallel group.
                current_name = (
                    all_step_names[idx + 1] if idx + 1 < len(all_step_names) else None
                )
                prev_env = last_ok or prev_env
                was_routed = False  # routing does not apply to parallel groups
                last_snap = self._save_checkpoint(
                    effective_key=effective_key,
                    last_snapshot=last_snap, next_step=current_name,
                    kv=kv, completed=completed,
                    status="running" if current_name else "done",
                    run_uid=run_uid,
                )
                continue

            # ──────────────────────────────────────────────────────────────
            # Sequential path
            # ──────────────────────────────────────────────────────────────
            result_env = await self._execute_one(
                step, prev_env, start_env, history, kv,
                tool_map=tool_map, session=session, run_id=run_id,
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
                return result_env

            # Persist writes
            if step.writes and result_env.payload is not None:
                kv[step.writes] = result_env.payload
                if effective_store:
                    effective_store.write(step.writes, result_env.payload)

            history.append(StepResult(step_name=step.name or "", envelope=result_env))
            completed.append(step.name or "")
            prev_env = result_env

            # Determine next step via out.next or linear progression
            next_name, next_was_routed = self._routing(result_env, step, step_map, was_routed=was_routed)
            current_name = next_name
            was_routed = next_was_routed

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

        return prev_env

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
            ctx_env = self._resolve_sentinel(step.context, prev_env, start_env, history, kv)
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
            step, step_env, tool_map=tool_map, session=session, run_id=run_id, branch_id=branch_id,
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
                task=prev.text(), context=prev.context, payload=prev.payload,
                metadata=prev.metadata, error=prev.error,
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
                        task=e.text(), context=e.context, payload=e.payload,
                        metadata=e.metadata, error=e.error,
                    )
            return start
        if isinstance(sentinel, _FromParallel):
            for r in reversed(history):
                if r.step_name == sentinel.name:
                    e = r.envelope
                    return Envelope(
                        task=e.text(), context=e.context, payload=e.payload,
                        metadata=e.metadata, error=e.error,
                    )
            return start
        if isinstance(sentinel, str):
            return Envelope(task=sentinel, payload=sentinel)
        return prev

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
                raw = await tool.run(**{"task": task_str} if "task" in tool.definition().parameters.get("properties", {}) else _first_arg_kwargs(tool, task_str))
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
        *,
        was_routed: bool = False,
    ) -> tuple[str | None, bool]:
        """Determine next step. Returns (next_step_name, is_routed).

        If the current step's output has a ``next`` field, follow it (explicit routing).
        If this step was itself reached via routing (was_routed=True), do NOT apply
        linear progression — only follow explicit out.next. This prevents sibling
        branches from running after one branch completes.
        """
        payload = result_env.payload
        if payload is not None and hasattr(payload, "next"):
            next_val = payload.next
            if isinstance(next_val, str) and next_val in step_map:
                return next_val, True  # routed explicitly

        # If we got here via explicit routing, stop — don't bleed into next linear step
        if was_routed:
            return None, False

        # Linear progression
        steps_list = list(step_map.keys())
        try:
            idx = steps_list.index(step.name or "")
            if idx + 1 < len(steps_list):
                return steps_list[idx + 1], False
        except ValueError:
            pass
        return None, False

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
    async def stream(self, env: Envelope, *, tools: list, output_type: type, memory: Any, session: Any) -> AsyncIterator[str]:
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
    if isinstance(sentinel, str):
        return {"kind": "literal", "value": sentinel}
    return None


def _sentinel_from_ref(ref: dict[str, Any] | None) -> Sentinel | str:
    if ref is None:
        return from_prev
    from lazybridge.sentinels import from_parallel, from_start, from_step

    kind = ref.get("kind")
    if kind == "from_prev":
        return from_prev
    if kind == "from_start":
        return from_start
    if kind == "from_step":
        return from_step(ref["name"])
    if kind == "from_parallel":
        return from_parallel(ref["name"])
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
        d["context"] = _sentinel_to_ref(step.context)
    if step.writes:
        d["writes"] = step.writes
    return d


def _step_from_dict(data: dict[str, Any], registry: dict[str, Any]) -> Step:
    target = _target_from_ref(data["target"], registry)
    task = _sentinel_from_ref(data.get("task"))
    context = _sentinel_from_ref(data["context"]) if "context" in data else None
    return Step(
        target=target,
        task=task,
        context=context,
        writes=data.get("writes"),
        parallel=data.get("parallel", False),
        name=data.get("name"),
    )
