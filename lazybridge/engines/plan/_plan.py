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
from typing import TYPE_CHECKING, Any, Literal, cast

from lazybridge.core.streaming import restore_ambient_token_sink, suppress_ambient_token_sink
from lazybridge.engines.base import resolve_agent_name
from lazybridge.engines.plan._checkpoint import _WRITE_STAMP_PREFIX, CheckpointMixin
from lazybridge.engines.plan._compiler import PlanCompiler
from lazybridge.engines.plan._fanout import FanoutMixin
from lazybridge.engines.plan._resolve import ResolveMixin
from lazybridge.engines.plan._serialisation import (
    _first_arg_kwargs,
    _step_from_dict,
    _step_to_dict,
)
from lazybridge.engines.plan._types import (
    ConcurrentPlanRunError,
    PlanPaused,
    PlanRuntimeError,
    PlanState,
    Step,
    StepResult,
)
from lazybridge.envelope import Envelope, EnvelopeMetadata, ErrorInfo
from lazybridge.session import EventType
from lazybridge.signals import ConcludeSignal

if TYPE_CHECKING:
    from lazybridge.memory import Memory
    from lazybridge.session import Session
    from lazybridge.store import Store
    from lazybridge.tools import Tool


class Plan(CheckpointMixin, ResolveMixin, FanoutMixin):
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
        stream_buffer: int = 64,
    ) -> None:
        """Construct a Plan.

        Streaming
        ---------
        ``stream()`` yields tokens live from each step's LLM engine as the
        plan executes (see :meth:`stream`).  ``stream_buffer`` bounds the
        token queue exactly like ``LLMEngine(stream_buffer=...)`` — a slow
        consumer throttles the producing step instead of growing memory.

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

        The persisted shape (v2) includes minimal plan state plus serialized
        StepResult history: ``{"next_step": str, "kv": {...},
        "completed_steps": [...], "status": str, "run_uid": str,
        "history": [...]}``.
        History is serialized so a resumed run can re-aggregate
        ``from_parallel_all`` bands and nested-cost rollup against completed
        upstream steps.  Only ``writes``-bucket values and step history
        survive across process boundaries; live in-memory state does not.

        Crash-window durability
        -----------------------
        Each step writes its checkpoint *before* the durable
        ``store.write(step.writes, value)`` call.  This eliminates
        double-writes on resume — the checkpoint already records
        ``next_step`` as the following step, so a resumed run does not
        re-execute the completed step.  The trade-off is that a crash in
        the gap between the checkpoint and the Store write makes the
        durable Store write *lost*; the value still lives in the
        checkpoint's serialised ``kv`` and is read back into in-memory
        state on resume, so the Plan continues correctly, but **sidecar
        consumers reading the Store directly should reconcile against
        the checkpoint snapshot rather than assume Store completeness**.

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
        if stream_buffer < 1:
            raise ValueError(f"stream_buffer must be >= 1, got {stream_buffer!r}")
        self.stream_buffer = stream_buffer
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
        result_env: Envelope[Any],
        history: list[StepResult],
    ) -> Envelope[Any]:
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
        return cast(Envelope[Any], result_env.model_copy(update={"metadata": new_meta}))

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    async def run(
        self,
        env: Envelope[Any],
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Store | None = None,
        plan_state: PlanState | None = None,
    ) -> Envelope[Any]:
        # ``Agent.__init__`` stamps the wrapping agent's name on the engine
        # via ``engine._agent_name = self.name``.  Plan emits AGENT_START
        # and AGENT_FINISH like every other engine so replay mode (which
        # rebuilds the graph from events alone) can see the parent node.
        run_id = str(uuid.uuid4())
        agent_name = resolve_agent_name(self, "plan")
        if session:
            session.emit(
                EventType.AGENT_START,
                {"agent_name": agent_name, "task": env.task},
                run_id=run_id,
            )

        result_env: Envelope[Any] | None = None
        error_msg: str | None = None
        try:
            result_env = await self._run_impl(
                env,
                tools=tools,
                output_type=output_type,
                memory=memory,
                session=session,
                store=store,
                plan_state=plan_state,
                run_id=run_id,
                agent_name=agent_name,
            )
            return result_env
        except BaseException as exc:  # propagate after emitting AGENT_FINISH
            error_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if session:
                payload: dict[str, Any] = {"agent_name": agent_name}
                if error_msg is not None:
                    payload["error"] = error_msg
                elif result_env is not None and result_env.error is not None:
                    payload["error"] = result_env.error.message
                elif result_env is not None:
                    payload["payload"] = (result_env.text() or "")[:500]
                session.emit(EventType.AGENT_FINISH, payload, run_id=run_id)

    async def _run_impl(
        self,
        env: Envelope[Any],
        *,
        tools: list[Tool],
        output_type: type,
        memory: Memory | None,
        session: Session | None,
        store: Store | None = None,
        plan_state: PlanState | None = None,
        run_id: str,
        agent_name: str,
    ) -> Envelope[Any]:
        # ``run_id`` and ``agent_name`` are threaded in from ``run()`` so
        # AGENT_START/FINISH and TOOL_CALL/RESULT all share the same
        # correlation id — matches how LLMEngine and HumanEngine attach
        # their child events to the agent span.
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
        # Serialised twin of ``history``, maintained incrementally so each
        # checkpoint save reuses prior serialisation work instead of
        # re-dumping every envelope (O(n²) across a long plan).
        history_payload: list[dict[str, Any]] = self._history_to_payload(history) if history else []
        kv: dict[str, Any] = dict(plan_state.store) if plan_state else {}
        completed: list[str] = []
        # The effective durable store: per-call ``store=`` wins over the
        # Plan-level one.  Computed once — used by both the resume replay
        # below and the step-write path in the main loop.
        effective_store = store if store is not None else self.store
        if checkpoint is not None:
            kv.update(checkpoint.get("kv") or {})
            completed = list(checkpoint.get("completed_steps") or [])
            # v2+ checkpoints carry the step-result history so a resumed
            # run can re-aggregate ``from_parallel_all`` bands and the
            # nested-cost rollup against upstream completed steps.  v1
            # checkpoints (no ``history`` key) degrade to an empty
            # in-memory history — same behaviour as pre-W1.3, no crash.
            restored_payload = checkpoint.get("history") or []
            history.extend(self._payload_to_history(restored_payload))
            if isinstance(restored_payload, list):
                history_payload.extend(d for d in restored_payload if isinstance(d, dict))
            # Replay Store sidecar writes for any step that completed but
            # whose durable write was lost in the checkpoint→Store gap on
            # the prior run.  Idempotent: writing the same value is a no-op
            # for any sane Store backend.  Closes the "external consumers
            # see incomplete state" failure mode.
            if effective_store is not None:
                completed_set = set(completed)
                for step_def in self.steps:
                    if step_def.name in completed_set and step_def.writes and step_def.writes in kv:
                        effective_store.write(
                            step_def.writes, kv[step_def.writes], agent_id=_WRITE_STAMP_PREFIX + run_uid
                        )

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
        # Maps branch-step name → after_branches rejoin target.  Populated
        # when a routing step with after_branches fires; consumed when the
        # branch step completes so _routing() can jump to the rejoin point.
        branch_return: dict[str, str] = {}

        all_step_names = [s.name for s in self.steps]

        # Non-local exits (cancellation, conclude, unexpected exceptions)
        # unwind past the normal per-step checkpointing.  Without a
        # terminal checkpoint the key stays claimed by a dead run_uid and
        # every later on_concurrent='fail' run raises
        # ConcurrentPlanRunError — e.g. a consumer breaking out of
        # plan.stream() early poisoned the key permanently.
        try:
            while current_name and iterations < self.max_iterations:
                iterations += 1
                step = step_map.get(current_name)
                if not step:
                    break

                # Parallel branch dispatch — see _run_parallel_band.
                if step.parallel:
                    terminal_env, current_name, prev_env, last_snap = await self._run_parallel_band(
                        step,
                        all_step_names=all_step_names,
                        step_map=step_map,
                        prev_env=prev_env,
                        start_env=start_env,
                        history=history,
                        history_payload=history_payload,
                        kv=kv,
                        completed=completed,
                        tool_map=tool_map,
                        session=session,
                        run_id=run_id,
                        agent_name=agent_name,
                        effective_key=effective_key,
                        effective_store=effective_store,
                        last_snap=last_snap,
                        run_uid=run_uid,
                    )
                    if terminal_env is not None:
                        return terminal_env
                    continue

                # ──────────────────────────────────────────────────────────────
                # Sequential path
                # ──────────────────────────────────────────────────────────────
                try:
                    result_env = await self._execute_one(
                        step,
                        prev_env,
                        start_env,
                        history,
                        kv,
                        tool_map=tool_map,
                        session=session,
                        run_id=run_id,
                        agent_name=agent_name,
                    )
                except PlanPaused as pause:
                    # Cooperative pause — write a paused checkpoint pointing
                    # back at this step (resume=True will re-invoke it) and
                    # return a paused-error envelope so the caller can detect.
                    last_snap = self._save_checkpoint(
                        effective_key=effective_key,
                        last_snapshot=last_snap,
                        next_step=step.name,
                        kv=kv,
                        completed=completed,
                        status="paused",
                        run_uid=run_uid,
                        history_payload=history_payload,
                    )
                    paused_env = Envelope(
                        task=prev_env.task,
                        context=prev_env.context,
                        payload=prev_env.payload,
                        metadata=prev_env.metadata,
                        error=ErrorInfo(
                            type="PlanPaused",
                            message=f"Plan paused at step {step.name!r}: {pause.message}",
                            retryable=True,
                        ),
                    )
                    return self._aggregate_nested_metadata(paused_env, history)

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
                        history_payload=history_payload,
                    )
                    return self._aggregate_nested_metadata(result_env, history)

                # Persist writes — in-memory only here so kv goes into the checkpoint below.
                if step.writes and result_env.payload is not None:
                    kv[step.writes] = result_env.payload

                seq_sr = StepResult(step_name=step.name or "", envelope=result_env)
                history.append(seq_sr)
                history_payload.extend(self._history_to_payload([seq_sr]))
                completed.append(step.name or "")
                prev_env = result_env

                # Determine next step via routes / routes_by or linear progression.
                current_name = self._routing(result_env, step, step_map, branch_return)

                # Save checkpoint first; durable store write comes after so a crash
                # between the two is safe — resume sees the checkpoint and skips the
                # step rather than re-running it (no double-write).  Inverse trade:
                # a crash in the gap means the durable Store write is *lost*; the
                # value still survives in the checkpoint's ``kv`` for Plan replay,
                # so any sidecar that reads the Store directly should reconcile
                # against the checkpoint snapshot, not assume Store completeness.
                last_snap = self._save_checkpoint(
                    effective_key=effective_key,
                    last_snapshot=last_snap,
                    next_step=current_name,
                    kv=kv,
                    completed=completed,
                    status="running" if current_name else "done",
                    run_uid=run_uid,
                    history_payload=history_payload,
                )
                if step.writes and result_env.payload is not None and effective_store:
                    effective_store.write(step.writes, result_env.payload, agent_id=_WRITE_STAMP_PREFIX + run_uid)
        except ConcludeSignal:
            # conclude() ends the task with an answer — from the
            # checkpoint's perspective the plan is finished.
            self._terminal_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snap,
                next_step=None,
                kv=kv,
                completed=completed,
                status="done",
                run_uid=run_uid,
                history_payload=history_payload,
            )
            raise
        except asyncio.CancelledError:
            # Cancelled mid-flight (stream consumer disconnected, outer
            # timeout).  Record a terminal 'cancelled' state pointing at
            # the incomplete step: a fresh run can claim over it, and
            # resume=True continues from next_step.
            self._terminal_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snap,
                next_step=current_name,
                kv=kv,
                completed=completed,
                status="cancelled",
                run_uid=run_uid,
                history_payload=history_payload,
            )
            raise
        except ConcurrentPlanRunError:
            # We lost the key to another writer — it is theirs now; a
            # terminal write would clobber their state.
            raise
        except Exception:
            # Unexpected failure escaping the step machinery (e.g. a
            # PlanRuntimeError from sentinel resolution).  Mark failed at
            # the current step so resume=True can retry it.
            self._terminal_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snap,
                next_step=current_name,
                kv=kv,
                completed=completed,
                status="failed",
                run_uid=run_uid,
                history_payload=history_payload,
            )
            raise

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

    async def _run_parallel_band(
        self,
        step: Step,
        *,
        all_step_names: list[str | None],
        step_map: dict[str, Step],
        prev_env: Envelope[Any],
        start_env: Envelope[Any],
        history: list[StepResult],
        history_payload: list[dict[str, Any]],
        kv: dict[str, Any],
        completed: list[str],
        tool_map: dict[str, Tool],
        session: Session | None,
        run_id: str,
        agent_name: str,
        effective_key: str | None,
        effective_store: Store | None,
        last_snap: dict[str, Any] | None,
        run_uid: str,
    ) -> tuple[Envelope[Any] | None, str | None, Envelope[Any], dict[str, Any] | None]:
        """Dispatch one contiguous ``parallel=True`` band and apply its state.

        Collects every consecutive ``parallel=True`` step from ``step`` in
        DECLARED order and runs them concurrently via ``asyncio.gather``.
        Each branch sees the SAME ``prev_env`` / ``history`` / ``kv``
        snapshot — a branch cannot observe its siblings' effects.  State
        updates apply sequentially after the gather so ``writes`` are
        deterministic.  Routing (``routes`` / ``routes_by``) is ignored on
        parallel branches — control flow after the band falls through to
        the next declared step in linear order (the conventional "join").

        Mutates ``history`` / ``history_payload`` / ``kv`` / ``completed``
        in place (all-success path only — atomicity on failure/pause).
        Returns ``(terminal_env, next_name, prev_env, last_snap)``:
        ``terminal_env`` is non-``None`` when the band paused or failed
        (the caller returns it as the run result); otherwise the loop
        continues at ``next_name`` with the updated ``prev_env`` /
        ``last_snap``.  A ``ConcludeSignal`` from any branch propagates.
        """
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
        # Streaming: unbind the ambient token sink around the band —
        # gather's tasks snapshot the suppressed context, so
        # concurrent branches cannot interleave tokens (see
        # lazybridge/core/streaming.py).
        _sink_suppress = suppress_ambient_token_sink()
        try:
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
                        agent_name=agent_name,
                    )
                    for s in group
                ],
                return_exceptions=True,
            )
        finally:
            restore_ambient_token_sink(_sink_suppress)

        # Atomicity: scan ALL branches for failure first.  If any
        # branch errored we return WITHOUT applying any writes to
        # ``kv`` / ``effective_store`` / ``history`` / ``completed``
        # — so a later resume re-runs the whole band cleanly
        # instead of partially-double-applying side-effects from
        # siblings that succeeded earlier in the iteration order.
        # Cooperative pause: any branch raising PlanPaused halts
        # the whole band atomically.  Same atomicity as failure —
        # no writes from succeeded siblings are applied; resume
        # re-runs the whole band cleanly.
        # A ``conclude`` from any branch ends the whole task: re-raise
        # so it unwinds the plan to the top-level caller (the band's
        # other branch results are discarded), the same non-local exit
        # sequential steps get.
        for r in raw:
            if isinstance(r, ConcludeSignal):
                raise r

        paused_branch: tuple[Step, PlanPaused] | None = None
        for _s, r in zip(group, raw):
            if isinstance(r, PlanPaused):
                paused_branch = (_s, r)
                break
        if paused_branch is not None:
            last_snap = self._save_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snap,
                next_step=group[0].name,  # whole band re-runs on resume
                kv=kv,
                completed=completed,
                status="paused",
                run_uid=run_uid,
                history_payload=history_payload,
            )
            paused_env: Envelope[Any] = Envelope(
                task=prev_env.task,
                context=prev_env.context,
                payload=prev_env.payload,
                metadata=prev_env.metadata,
                error=ErrorInfo(
                    type="PlanPaused",
                    message=(f"Plan paused at parallel step {paused_branch[0].name!r}: {paused_branch[1].message}"),
                    retryable=True,
                ),
            )
            return self._aggregate_nested_metadata(paused_env, history), None, prev_env, last_snap

        first_failure_env: Envelope[Any] | None = None
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
                history_payload=history_payload,
            )
            return self._aggregate_nested_metadata(first_failure_env, history), None, prev_env, last_snap

        # All branches succeeded — collect results, update in-memory kv,
        # checkpoint FIRST, then flush to the durable store.  This ordering
        # ensures that a crash between the checkpoint and the store.write()
        # does NOT replay the step on resume (the checkpoint already records
        # next_step as the step after this band), so no double-write occurs.
        #
        # Trade-off: the inverse failure mode is a *lost durable write* —
        # if we crash between checkpoint and store.write, the resumed run
        # advances past the band and never retries the durable write.
        # The values still live in the checkpoint's serialised ``kv``, so
        # Plan replay reads them correctly via ``_load_checkpoint``; only
        # sidecar consumers reading the Store directly would observe the
        # gap.  See ``Plan.run`` doc above for the full contract.
        last_ok: Envelope[Any] | None = None
        for s, r in zip(group, raw):
            step_name = s.name or ""
            # Type-narrow: failure-scan above already returned on
            # BaseException / r.error; remaining ``r`` are Envelopes.
            assert isinstance(r, Envelope)
            if s.writes and r.payload is not None:
                kv[s.writes] = r.payload  # in-memory; goes into checkpoint below
            sr = StepResult(step_name=step_name, envelope=r)
            history.append(sr)
            history_payload.extend(self._history_to_payload([sr]))
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
            history_payload=history_payload,
        )
        # Durable store writes AFTER checkpoint — crash here is safe because
        # the checkpoint already points to the next step.
        if effective_store:
            for s, r in zip(group, raw):
                # Same narrowing as the loop at line 706: the
                # failure scan above already returned on
                # BaseException, so every remaining ``r`` is an
                # Envelope.  Repeat the assert to keep mypy happy
                # in this scope.
                assert isinstance(r, Envelope)
                if s.writes and r.payload is not None:
                    effective_store.write(s.writes, r.payload, agent_id=_WRITE_STAMP_PREFIX + run_uid)
        return None, current_name, prev_env, last_snap

    async def _execute_one(
        self,
        step: Step,
        prev_env: Envelope[Any],
        start_env: Envelope[Any],
        history: list[StepResult],
        kv: dict[str, Any],
        *,
        tool_map: dict[str, Tool],
        session: Session | None,
        run_id: str,
        branch_id: str | None = None,
        agent_name: str | None = None,
    ) -> Envelope[Any]:
        """Resolve sentinels, build the step env, and execute the step.

        Returns a normalised ``Envelope``.  Does NOT mutate ``history`` /
        ``kv`` — the caller applies those deterministically so parallel
        branches see a consistent snapshot.
        ``branch_id`` is set for parallel-branch steps so their Session
        events can be distinguished from sequential-step events.
        """
        step_task_env = self._resolve_sentinel(step.task, prev_env, start_env, history, kv, tool_map)

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
                ctx_env = self._resolve_sentinel(item, prev_env, start_env, history, kv, tool_map)
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
            agent_name=agent_name,
        )
        return Envelope(
            task=step_env.task,
            context=step_env.context,
            payload=result_env if not isinstance(result_env, Envelope) else result_env.payload,
            metadata=result_env.metadata if isinstance(result_env, Envelope) else EnvelopeMetadata(),
            error=result_env.error if isinstance(result_env, Envelope) else None,
        )

    async def _exec_step(
        self,
        step: Step,
        env: Envelope[Any],
        *,
        tool_map: dict[str, Tool],
        session: Session | None,
        run_id: str,
        branch_id: str | None = None,
        agent_name: str | None = None,
    ) -> Envelope[Any]:
        if session:
            payload: dict[str, Any] = {"step": step.name, "task": env.task}
            if branch_id is not None:
                payload["branch_id"] = branch_id
            if agent_name is not None:
                # Tag every TOOL_CALL with the wrapping agent so replay
                # can rebuild the parent → step edge from events alone.
                payload["agent_name"] = agent_name
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
                # Agent as step.  Use ``_run_as_tool`` (not ``run``) so a
                # ``conclude`` raised inside the step propagates out of the plan
                # to the top-level caller, matching the tool-target path above.
                # Duck-typed doubles without it fall back to ``run``.
                runner = getattr(target, "_run_as_tool", target.run)
                result_env = await runner(env)
            elif callable(target):
                # Raw callable.  Sync targets are dispatched to the default
                # executor so blocking I/O / CPU work in a user-supplied
                # function does not stall the event loop.  This mirrors the
                # contract Tool.run() (lazybridge/tools.py) already
                # enforces for tools registered via ``Tool(fn)``.
                #
                # ``contextvars.copy_context().run`` propagates the
                # caller's contextvars into the executor thread so
                # request IDs, OpenTelemetry spans, and any other
                # context-local observability set on the calling
                # coroutine remain visible inside ``target``.  Without
                # this wrap, the default executor's empty thread
                # context silently drops those values.
                import asyncio as _asyncio
                import contextvars as _contextvars
                import inspect as _inspect

                arg = env.task or env.text()
                if _inspect.iscoroutinefunction(target):
                    raw = await target(arg)
                else:
                    loop = _asyncio.get_running_loop()
                    ctx = _contextvars.copy_context()
                    raw = await loop.run_in_executor(None, ctx.run, target, arg)
                result_env = Envelope(task=env.task, payload=raw)
            else:
                raise RuntimeError(f"Cannot execute step target: {target!r}")

            if session:
                result_payload: dict[str, Any] = {"step": step.name, "result": result_env.text()[:200]}
                if branch_id is not None:
                    result_payload["branch_id"] = branch_id
                if agent_name is not None:
                    result_payload["agent_name"] = agent_name
                session.emit(EventType.TOOL_RESULT, result_payload, run_id=run_id)
            return result_env

        except Exception as exc:
            if session:
                err_payload: dict[str, Any] = {"step": step.name, "error": str(exc)}
                if branch_id is not None:
                    err_payload["branch_id"] = branch_id
                if agent_name is not None:
                    err_payload["agent_name"] = agent_name
                session.emit(EventType.TOOL_ERROR, err_payload, run_id=run_id)
            return Envelope.error_envelope(exc)

    def _routing(
        self,
        result_env: Envelope[Any],
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
                    # A misbehaving predicate is a bug, not a recoverable
                    # runtime condition — surface it instead of silently
                    # falling through to linear progression and masking
                    # the failure.  ``PlanRuntimeError`` (not
                    # ``PlanCompileError``) is the right class because
                    # this fires at run time, not at Plan construction;
                    # mixing the two would force callers to catch
                    # ``PlanCompileError`` at runtime, conflating
                    # build-time and runtime failure modes.
                    raise PlanRuntimeError(
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

        Callables, Agents, and ``output=`` / ``input=`` types are
        serialised by ``name`` only — rebind them at load time via
        :meth:`from_dict`'s ``registry`` kwarg (types under a
        ``"type:<Name>"`` or bare ``"<Name>"`` key).  Sentinels, writes,
        parallel flags, iteration limit, and step order are preserved
        faithfully.

        Version history: v1 omitted step ``output`` / ``input`` — every
        structured step silently degraded to ``output=str`` on reload
        and ``routes_by`` plans failed to recompile.  v2 records both.
        v1 payloads still load (missing keys default to ``str`` /
        ``Any``).
        """
        return {
            "version": 2,
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
        self, env: Envelope[Any], *, tools: list[Any], output_type: type, memory: Any, session: Any
    ) -> AsyncIterator[str]:
        """Stream tokens live from the plan's steps as they execute.

        The plan runs exactly as :meth:`run` does (same checkpointing,
        routing, and metadata aggregation); while it runs, every
        *sequential* step whose target bottoms out in an ``LLMEngine``
        streams its tokens here as they are generated — so the consumer
        watches the pipeline think step by step instead of waiting for one
        final text block.

        Semantics:

        * **Parallel bands do not stream** — concurrent branches would
          interleave tokens into an unreadable shuffle; their results still
          flow into downstream steps, whose tokens do stream.
        * **Nested tool-calls are silent** — within a step, agents invoked
          as tools between turns do not stream, matching
          ``LLMEngine.stream()``.
        * **Fallback** — a plan with no streaming-capable step (pure
          functions, mock engines) yields the final text once, preserving
          the pre-streaming contract.
        * **Verification caveat** — as with ``Agent.stream()``, tokens are
          emitted before any ``verify=`` / ``output=`` post-processing on
          the step's agent completes.
        * **Store output caveat** — ``Agent.stream()`` persists the joined
          streamed tokens under the agent's output key.  For a plan engine
          that is the concatenated narration of every streamed step, not
          the final step's text alone (which is what ``Agent.run()``
          stores).  Downstream consumers needing exact step values should
          read the ``Step(writes=...)`` keys instead.
        """
        from lazybridge.core.streaming import stream_envelope_run

        async def _run() -> Envelope[Any]:
            return await self.run(env, tools=tools, output_type=output_type, memory=memory, session=session)

        async for token in stream_envelope_run(_run, buffer=self.stream_buffer):
            yield token
