"""Plan — multi-step structured execution engine.

The bulk of the runtime: ``__init__`` validation, async ``run``,
parallel-band scheduling, sentinel resolution, routing, checkpoint
write/load/claim, history aggregation, ``run_many`` / ``arun_many``
fan-out, and ``to_dict`` / ``from_dict`` serialisation.

Carved out of the old monolithic ``plan.py`` (W3.1) — class body and
public method surface are unchanged; structural validation has moved
to :mod:`._compiler`, dataclasses to :mod:`._types`, and serialisation
helpers to :mod:`._serialisation`.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

from lazybridge.engines.plan._compiler import PlanCompiler
from lazybridge.engines.plan._serialisation import (
    _first_arg_kwargs,
    _step_from_dict,
    _step_to_dict,
)
from lazybridge.engines.plan._types import (
    ConcurrentPlanRunError,
    PlanCompileError,
    PlanState,
    Step,
    StepResult,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.sentinels import (
    Sentinel,
    _FromParallel,
    _FromParallelAll,
    _FromPrev,
    _FromStart,
    _FromStep,
)
from lazybridge.session import EventType

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.store import Store
    from lazybridge.tools import Tool


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

    #: Checkpoint schema version.  Bumped when the on-disk layout
    #: changes in a non-additive way.  v1 = no ``history`` key (pre-W1.3);
    #: v2 = adds ``history`` (serialised StepResult list) so resume can
    #: re-aggregate ``from_parallel_all`` and nested-cost rollup against
    #: completed upstream steps.  Older checkpoints are read as v1 with
    #: an empty in-memory ``history`` — degrades to pre-W1.3 behaviour
    #: (the parallel band aggregator falls back to ``start_env``) without
    #: crashing.
    CHECKPOINT_VERSION: int = 2

    @staticmethod
    def _history_to_payload(history: list[StepResult]) -> list[dict[str, Any]]:
        """JSON-friendly serialisation of the step-result history.

        ``Envelope`` is a Pydantic model; ``model_dump(mode="json")``
        produces a JSON-compatible dict.  Non-Pydantic payloads fall
        through to ``str`` via Pydantic's default serialisation (best
        effort).  On reload we accept whatever shape comes back —
        ``Envelope.text()`` is JSON-aware so ``from_parallel_all``
        renders correctly regardless of the original payload type.
        """
        out: list[dict[str, Any]] = []
        for sr in history:
            try:
                env_dump = sr.envelope.model_dump(mode="json")
            except Exception:
                # Best-effort: a payload that isn't Pydantic/JSON-clean
                # falls back to its string form; the envelope's
                # metadata, task, and error survive.
                env_dump = {
                    "task": sr.envelope.task,
                    "context": sr.envelope.context,
                    "payload": str(sr.envelope.payload) if sr.envelope.payload is not None else None,
                    "metadata": sr.envelope.metadata.model_dump(mode="json"),
                    "error": sr.envelope.error.model_dump(mode="json") if sr.envelope.error else None,
                }
            out.append({"step_name": sr.step_name, "envelope": env_dump, "ts": sr.ts})
        return out

    @staticmethod
    def _payload_to_history(data: Any) -> list[StepResult]:
        """Inverse of :meth:`_history_to_payload`.  Tolerant of missing
        / malformed entries — drops them silently rather than failing
        the whole resume.
        """
        if not isinstance(data, list):
            return []
        out: list[StepResult] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            env_data = item.get("envelope")
            if not isinstance(env_data, dict):
                continue
            try:
                env: Envelope = Envelope.model_validate(env_data)
            except Exception:
                continue
            out.append(
                StepResult(
                    step_name=str(item.get("step_name") or ""),
                    envelope=env,
                    ts=float(item.get("ts") or 0.0),
                )
            )
        return out

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
        history: list[StepResult] | None = None,
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

        ``history`` is the live in-memory step-result list at write
        time; persisting it lets a resumed run re-aggregate
        ``from_parallel_all`` and nested-cost rollup against upstream
        steps that completed before the crash.
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
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "history": self._history_to_payload(history) if history else [],
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
        claimed_snap: dict[str, Any] = {
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
            # v2+ checkpoints carry the step-result history so a resumed
            # run can re-aggregate ``from_parallel_all`` bands and the
            # nested-cost rollup against upstream completed steps.  v1
            # checkpoints (no ``history`` key) degrade to an empty
            # in-memory history — same behaviour as pre-W1.3, no crash.
            history.extend(self._payload_to_history(checkpoint.get("history") or []))

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
        # Maps branch-step name → after_branches rejoin target.  Populated
        # when a routing step with after_branches fires; consumed when the
        # branch step completes so _routing() can jump to the rejoin point.
        branch_return: dict[str, str] = {}

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
                first_failure_env: Envelope | None = None
                for _s, r in zip(group, raw):
                    if isinstance(r, BaseException):
                        first_failure_env = Envelope.error_envelope(r)
                        break
                    if r.error is not None:
                        first_failure_env = r
                        break
                if first_failure_env is not None:
                    # Point the checkpoint at the band's FIRST step, not the
                    # failing step.  On resume the whole band must re-run so
                    # all siblings produce fresh writes; resuming mid-band
                    # would silently skip earlier siblings and leave kv stale.
                    last_snap = self._save_checkpoint(
                        effective_key=effective_key,
                        last_snapshot=last_snap,
                        next_step=group[0].name,
                        kv=kv,
                        completed=completed,
                        status="failed",
                        run_uid=run_uid,
                        history=history,
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
                    history=history,
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
                    history=history,
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
            current_name = self._routing(result_env, step, step_map, branch_return)

            # Save checkpoint after each step so a crash mid-plan can resume
            last_snap = self._save_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snap,
                next_step=current_name,
                kv=kv,
                completed=completed,
                status="running" if current_name else "done",
                run_uid=run_uid,
                history=history,
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
            err_env: Envelope[Any] = Envelope(
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
        # Multimodal: thread attachments through so step 0 (or any step
        # whose ``task=`` resolves to ``_FromStart``) sees the original
        # images / audio.  Steps that resolve via ``_FromPrev`` /
        # ``_FromStep`` naturally inherit ``None`` because text-output
        # steps don't produce attachments — propagation is automatic.
        step_env: Envelope[Any] = Envelope(
            task=step_task_env.task,
            context=merged_ctx,
            images=step_task_env.images,
            audio=step_task_env.audio,
            payload=step_task_env.payload,
        )

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
            # Multimodal: preserve attachments — for step 0 ``prev`` is
            # the user-supplied input envelope and the user expects
            # images / audio to reach the first agent.  For step N>0
            # ``prev`` is an upstream LLM result which doesn't populate
            # attachments, so this defaults to ``None`` automatically.
            return Envelope(
                task=prev.text(),
                context=prev.context,
                images=prev.images,
                audio=prev.audio,
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
                        images=e.images,
                        audio=e.audio,
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
                        images=e.images,
                        audio=e.audio,
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
                    result_env: Envelope[Any] = Envelope(
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
        branch_return: dict[str, str],
    ) -> str | None:
        """Determine the next step's name, or ``None`` to end the Plan.

        Routing is **explicit and visible at the Step level**: ``routes=``
        (predicate map) or ``routes_by=`` (field name on the structured
        output) declares the branches.  Falls through to linear
        progression when no branch matches.

        When ``step.after_branches`` is set and a branch fires, the
        branch step's name is registered in ``branch_return`` so that
        when that branch step completes, execution jumps to the rejoin
        point instead of continuing linearly through sibling branches.

        Without ``after_branches``, routing is a *detour*: after the
        routed-to step, linear progression resumes from its declared
        position (legacy behaviour, preserved for backward compat).
        """
        # 1. Predicate-based routing.  First matching predicate wins.
        if step.routes:
            for target_name, predicate in step.routes.items():
                try:
                    if predicate(result_env):
                        if step.after_branches:
                            branch_return[target_name] = step.after_branches
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
                    if step.after_branches:
                        branch_return[value] = step.after_branches
                    return value
                # ``None`` and any other non-matching value fall through
                # to linear — the model "decided not to route".

        # 3. Exclusive-branch return: this step was entered via a routing
        #    step that declared after_branches.  Jump to the rejoin point
        #    instead of continuing linearly through sibling branches.
        name = step.name or ""
        if name in branch_return:
            return branch_return.pop(name)

        # 4. Linear progression — next declared step, or end of plan.
        steps_list = list(step_map.keys())
        try:
            idx = steps_list.index(name)
            if idx + 1 < len(steps_list):
                return steps_list[idx + 1]
        except ValueError:
            pass
        return None

    # ------------------------------------------------------------------
    # run_many — concurrent fan-out over N inputs
    # ------------------------------------------------------------------
    #
    # Replaces the boilerplate
    #
    #     async def run_one(t): return await plan.run(Envelope(task=t), ...)
    #     def driver(t): return asyncio.run(run_one(t))
    #     with ThreadPoolExecutor(max_workers=N) as pool:
    #         results = list(pool.map(driver, tasks))
    #
    # with a single declarative call.  The "concurrent fan-out" pattern
    # really wants one of two shapes — fork-isolated runs (different
    # ``checkpoint_key`` per run) or default single-key runs — and the
    # Plan already knows which it is via ``on_concurrent``.  ``run_many``
    # picks the right asyncio shape on the caller's behalf.

    def run_many(
        self,
        tasks: list[str | Envelope],
        *,
        concurrency: int | None = None,
    ) -> list[Envelope]:
        """Run this Plan concurrently against ``N`` inputs; sync return.

        Each ``task`` is dispatched as its own ``Plan.run`` invocation
        on a fresh asyncio task; results are returned as a list in
        input order.  Pair with ``Plan(on_concurrent="fork", ...)`` for
        true fan-out workflows where each input claims its own
        per-run keyspace.

        Errors are returned as error envelopes in the corresponding
        slot — the call never raises (matches ``Agent.parallel``
        semantics).

        ``concurrency`` caps the number of in-flight runs via an
        asyncio semaphore.  ``None`` (default) lets every task fire
        immediately.

        See :meth:`arun_many` for the async variant when the caller is
        already inside an event loop.
        """
        # Re-use the sync-bridge that ``Agent.__call__`` ships with —
        # it propagates contextvars (OTel spans, request ids, …) into
        # the worker loop so observability flows through fan-outs.
        from lazybridge.agent import _run_coro_with_context

        return _run_coro_with_context(self.arun_many(tasks, concurrency=concurrency))

    async def arun_many(
        self,
        tasks: list[str | Envelope],
        *,
        concurrency: int | None = None,
    ) -> list[Envelope]:
        """Async counterpart to :meth:`run_many`.

        Use this directly when you're already inside an event loop and
        want to ``await`` the fan-out without the sync-bridge overhead.
        """
        sem = asyncio.Semaphore(concurrency) if concurrency else None

        async def _one(task: str | Envelope) -> Envelope:
            # ``Envelope.from_task`` populates BOTH ``task`` and
            # ``payload`` so the first step's ``from_prev`` resolves to
            # the user's input rather than an empty string.
            env = task if isinstance(task, Envelope) else Envelope.from_task(str(task))

            async def _go() -> Envelope:
                return await self.run(
                    env,
                    tools=[],
                    output_type=str,
                    memory=None,
                    session=None,
                )

            if sem is None:
                return await _go()
            async with sem:
                return await _go()

        raw = await asyncio.gather(
            *[_one(t) for t in tasks],
            return_exceptions=True,
        )
        # Wrap raised exceptions as error envelopes so the contract is
        # "list of envelopes in input order".  Plan.run normally
        # returns an error envelope itself, so this branch only fires
        # for genuine framework bugs / cancellations.
        return [
            r
            if isinstance(r, Envelope)
            else Envelope.error_envelope(r if isinstance(r, BaseException) else RuntimeError(str(r)))
            for r in raw
        ]

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
