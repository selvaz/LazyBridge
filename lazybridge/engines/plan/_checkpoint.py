"""Checkpoint / resume machinery for :class:`Plan` — the CAS state machine.

Carved out of ``_plan.py`` in the v1-stabilization refactor.  Owns the
full checkpoint lifecycle: claim (CAS ownership), save (CAS-guarded
per-step snapshots), load (resume), terminal writes on non-local exits,
and the (de)serialisation of the step-result history.  Behaviour is
unchanged — ``Plan`` inherits this mixin, so every method keeps its
original name and signature.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from lazybridge._process_owner import current_owner_fields, owner_is_alive, record_age
from lazybridge.engines.plan._types import ConcurrentPlanRunError, StepResult
from lazybridge.envelope import Envelope

if TYPE_CHECKING:
    from lazybridge.engines.plan._types import Step
    from lazybridge.store import Store

#: ``agent_id`` stamp prefix on every durable ``Step(writes=...)`` Store
#: write.  The full stamp is ``f"{_WRITE_STAMP_PREFIX}{run_uid}"`` where
#: ``run_uid`` is the same value persisted in the checkpoint snapshot —
#: which is what lets a sidecar consumer detect the crash-window staleness
#: documented in ``Plan.__init__`` (see ``Plan.store_write_is_current``).
_WRITE_STAMP_PREFIX = "plan-run:"
_ACTIVE_CHECKPOINT_RUNS: set[str] = set()
_logger = logging.getLogger(__name__)


class CheckpointMixin:
    """Checkpoint state machine shared into :class:`Plan` by inheritance.

    Expects the host class to provide ``self.store``,
    ``self.checkpoint_key``, ``self.resume``, ``self.on_concurrent``,
    and ``self.steps``.
    """

    # Attributes provided by the host class (Plan.__init__).
    store: Store | None
    checkpoint_key: str | None
    resume: bool
    on_concurrent: str
    steps: list[Step]
    adopt_unowned: bool
    owner_stale_after: float

    @staticmethod
    def store_write_is_current(store: Store, *, checkpoint_key: str, key: str) -> bool:
        """Can a sidecar consumer trust ``store[key]`` right now?

        Closes the documented crash-window blind spot: each step's
        checkpoint is written *before* its durable ``Step(writes=...)``
        Store write, so after a crash in that gap the checkpoint claims
        the step completed while ``store[key]`` still holds a stale value
        from an earlier run (or nothing).  Every plan Store write is
        therefore stamped with the owning run's ``run_uid``; this helper
        compares that stamp against the checkpoint's recorded ``run_uid``.

        Returns ``True`` only when ``key`` exists and was written by the
        run the checkpoint at ``checkpoint_key`` describes.  ``False``
        means "reconcile against the checkpoint's ``kv`` snapshot instead"
        (the value there is always consistent with ``completed_steps``).

        With ``on_concurrent="fork"``, pass the *effective* key
        (``f"{checkpoint_key}:{run_uid}"``) the run actually used.
        """
        snap = store.read(checkpoint_key)
        if not isinstance(snap, dict) or not snap.get("run_uid"):
            return False
        entry = store.read_entry(key)
        if entry is None:
            return False
        return entry.agent_id == _WRITE_STAMP_PREFIX + str(snap["run_uid"])

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
                env: Envelope[Any] = Envelope.model_validate(env_data)
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
        history_payload: list[dict[str, Any]] | None = None,
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

        ``history_payload`` is the *pre-serialised* step-result history
        (see :meth:`_history_to_payload`), maintained incrementally by
        ``_run_impl`` as steps complete.  Taking the serialised form —
        rather than re-dumping the whole ``list[StepResult]`` on every
        save — keeps per-step checkpoint cost from growing quadratically
        with plan length.  Persisting it lets a resumed run re-aggregate
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
            "history": list(history_payload) if history_payload else [],
            "updated_at": time.time(),
            **current_owner_fields(),
        }
        if not store.compare_and_swap(effective_key, last_snapshot, new_snap):
            raise ConcurrentPlanRunError(
                f"Checkpoint {effective_key!r} was modified by another "
                f"writer mid-run (our run_uid={run_uid!r}).  Two Plan runs "
                f"appear to share this key — use a unique checkpoint_key "
                f"per concurrent run, or pass on_concurrent='fork'."
            )
        return new_snap

    def _terminal_checkpoint(
        self,
        *,
        effective_key: str | None,
        last_snapshot: dict[str, Any] | None,
        next_step: str | None,
        kv: dict[str, Any],
        completed: list[str],
        status: str,
        run_uid: str,
        history_payload: list[dict[str, Any]] | None,
    ) -> None:
        """Best-effort terminal checkpoint on a non-local exit.

        Used when the run is unwound by something other than a normal
        return — cancellation (a ``stream()`` consumer disconnecting),
        a ``conclude()`` signal, or an unexpected exception.  Without
        this, the key stays ``claimed``/``running`` under a dead
        ``run_uid`` and every subsequent ``on_concurrent="fail"`` run
        raises :class:`ConcurrentPlanRunError` until the key is
        manually cleared.  Failures here are swallowed: the exit that
        triggered us must propagate unmasked, and a CAS loss simply
        means another run already took over the key.
        """
        try:
            self._save_checkpoint(
                effective_key=effective_key,
                last_snapshot=last_snapshot,
                next_step=next_step,
                kv=kv,
                completed=completed,
                status=status,
                run_uid=run_uid,
                history_payload=history_payload,
            )
        except Exception:
            # Deliberately swallowed: the non-local exit that triggered this
            # terminal write must propagate unmasked, and a CAS loss just
            # means another run legitimately owns the key now.
            pass

    def clear_checkpoint(self, *, uncertain_only: bool = True) -> bool:
        """Clear this Plan's checkpoint so an uncertain run can be reconciled.

        By default only an ``uncertain`` checkpoint is removed. Pass
        ``uncertain_only=False`` only after separately confirming no worker owns it.
        """
        store = self._checkpoint_store()
        if store is None or self.checkpoint_key is None:
            return False
        current = store.read(self.checkpoint_key)
        if not isinstance(current, dict) or (uncertain_only and current.get("status") != "uncertain"):
            return False
        return store.compare_and_swap(
            self.checkpoint_key, current, {**current, "status": "reconciled", "updated_at": time.time()}
        )

    def _claim_checkpoint(
        self,
        effective_key: str | None,
        run_uid: str,
        *,
        reuse_done: bool = True,
    ) -> dict[str, Any] | None:
        """Acquire ownership of ``checkpoint_key`` for this run.

        * Fresh run, key absent or a prior ``status=="done"`` checkpoint
          → CAS-write a ``status="claimed"`` placeholder up-front so
          two concurrent fresh runs collide here, before either has
          executed any step.
        * ``resume=True`` can adopt an inactive checkpoint under a new
          ``run_uid``. A live or unverified active owner blocks adoption.
        * In-flight checkpoint and ``resume=False`` → raise
          :class:`ConcurrentPlanRunError`.
        """
        store = self._checkpoint_store()
        if store is None or effective_key is None:
            return None
        existing = store.read(effective_key)
        owner_fields = current_owner_fields()
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
            "updated_at": time.time(),
            **owner_fields,
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
        if status == "uncertain":
            raise ConcurrentPlanRunError(f"Checkpoint {effective_key!r} requires reconciliation before replay.")
        if status == "done":
            # Store-backed resume returns the cached result. A fresh run or
            # explicit PlanState acquires a new claim before executing.
            if self.resume and reuse_done:
                return existing
            if not store.compare_and_swap(effective_key, existing, claimed_snap):
                raise ConcurrentPlanRunError(
                    f"Lost race claiming completed key {effective_key!r} — "
                    f"another run moved past 'done' before we could claim. "
                    f"Retry."
                )
            return claimed_snap
        if status == "cancelled" and not self.resume:
            # Terminal state left by a prior run that was cancelled
            # mid-flight (e.g. its ``stream()`` consumer disconnected).
            # Unlike "claimed"/"running" there is no live owner, so a
            # fresh run claims over it; ``resume=True`` falls through to
            # the adopt path below and continues from the recorded
            # ``next_step``.
            if not store.compare_and_swap(effective_key, existing, claimed_snap):
                raise ConcurrentPlanRunError(
                    f"Lost race claiming cancelled key {effective_key!r} — "
                    f"another run claimed it between our read and CAS.  Retry."
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
        if status in ("claimed", "running"):
            alive = owner_is_alive(existing)
            if alive is None:
                age = record_age(existing)
                if age is None:
                    entry = store.read_entry(effective_key)
                    age = time.time() - entry.written_at if entry is not None else None
                if not self.adopt_unowned and (age is None or age < self.owner_stale_after):
                    raise ConcurrentPlanRunError(f"Checkpoint {effective_key!r} has an unverified active owner.")
                _logger.warning("Adopting unowned active checkpoint %r (age=%r seconds)", effective_key, age)
            previous_run = existing.get("run_uid")
            if alive and (
                existing.get("owner_pid") != owner_fields["owner_pid"]
                or not isinstance(previous_run, str)
                or previous_run in _ACTIVE_CHECKPOINT_RUNS
            ):
                raise ConcurrentPlanRunError(f"Checkpoint {effective_key!r} is still owned by a live run.")

        # A resumed failed or paused checkpoint also needs an active claim
        # before the first step runs. Keep its old status for input restoration.
        adopted = {
            **existing,
            **owner_fields,
            "resume_status": existing.get("resume_status", status),
            "status": "claimed",
            "run_uid": run_uid,
        }
        if not store.compare_and_swap(effective_key, existing, adopted):
            raise ConcurrentPlanRunError(
                f"Lost race claiming {effective_key!r} for resume — "
                f"another run stamped it between our read and claim.  Retry."
            )
        return adopted
