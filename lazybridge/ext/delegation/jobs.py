"""Durable job records and workspace-scoped session keys for delegation.

Promoted from LazyCEO's background delegation plumbing into an extension
under the core-vs-ext policy (``docs/guides/core-vs-ext.md``). The record
shape intentionally remains compatible with the source project while the
default Store namespaces are neutral; callers migrating an existing Store
can pass the old prefixes explicitly.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from lazybridge import Store, Tool

# Re-exported: callers imported current_owner_fields from this module before 1.8.12.
from lazybridge._process_owner import current_owner_fields as current_owner_fields
from lazybridge._process_owner import owner_is_alive, record_age

_logger = logging.getLogger(__name__)

#: Neutral default for the shared registry. A caller needing LazyCEO's old
#: on-disk layout can pass ``prefix="ceo:codex-job:"`` explicitly, just as
#: ApprovalQueue and DurableKnowledgeBase accept compatibility prefixes.
DEFAULT_JOB_PREFIX = "delegation:job:"

#: Neutral prefix for workspace-scoped resumable session ids. Existing
#: LazyCEO stores can opt back into ``"ceo:simple:session-id:"``.
DEFAULT_SESSION_KEY_PREFIX = "delegation:session:"

#: Fields ``JobRegistry.write`` owns the SHAPE of -- never accepted through
#: ``extra``, which exists for a caller's OWN fields, not to let a careless
#: caller silently clobber ``job_id``/``kind``/``objective``/``status`` out
#: from under every other reader of this registry (``find``/``checker_tool``/
#: ``result_tool``/``reclaim_interrupted`` all assume these four are present
#: and mean what this module says they mean).
_RESERVED_FIELDS = frozenset({"job_id", "kind", "objective", "status"})


def session_id_key(workspace_root: Path, *, prefix: str = DEFAULT_SESSION_KEY_PREFIX) -> str:
    """Return the stable Store key for a workspace's resumable session.

    This is parameterized by ``workspace_root``, not one fixed key, because
    a Claude Code SDK session created under one cwd cannot be resumed from a
    different one. The short hash keeps paths out of the Store namespace
    while remaining stable across restarts in the same workspace.

    Resolved to an absolute, normalized path before hashing -- an
    unresolved relative root would scope the key to its literal spelling
    rather than the actual workspace: ``Path("project")`` launched from
    two different parent directories would otherwise hash to the SAME
    key for two different working directories, and a relative vs.
    absolute spelling of the same directory would hash to two DIFFERENT
    keys for the same one. Either mistake resumes an incompatible session
    or fails to find the existing one. Found by Codex review before this
    ever shipped.
    """
    resolved = Path(workspace_root).resolve()
    return prefix + hashlib.sha256(str(resolved).encode()).hexdigest()[:16]


class JobRegistry:
    """Store-backed registry shared by every background delegation tool."""

    def __init__(self, store: Store, *, prefix: str = DEFAULT_JOB_PREFIX) -> None:
        self._store = store
        self._prefix = prefix

    def _key(self, job_id: str) -> str:
        return f"{self._prefix}{job_id}"

    def write(
        self,
        job_id: str,
        objective: str,
        *,
        tool_name: str,
        status: str,
        plan_id: str | None = None,
        task_index: int | None = None,
        plan_task_text: str | None = None,
        result: str | None = None,
        error: str | None = None,
        #: Best-effort identity of the engine/model/effort this job ran
        #: with -- read off the actual engine instance by the caller (see
        #: ``background._engine_identity``), never a value this method
        #: invents on its own.
        engine: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        #: True once the worker has actually started running (crossed from
        #: "approved/queued" to "doing real work") -- distinct from
        #: ``status`` because "running" already covers both: a job can be
        #: ``status="running"`` while still waiting on ``pre_confirm``.
        execution_started: bool | None = None,
        #: Set once a terminal result envelope is available; left unset
        #: (and therefore absent from the record) for as long as cost is
        #: genuinely unknown -- see ``cost_unknown``.
        cost_usd: float | None = None,
        #: True while a job may already be spending money with no result
        #: envelope yet to read a real cost from (set at dispatch, right
        #: before the worker starts); explicitly False once a terminal
        #: ``cost_usd`` has been recorded. Mirrors LazyCEO's
        #: ``cost_report.py`` convention: both fields together are what an
        #: unmeasured-cost accounting pass needs.
        cost_unknown: bool | None = None,
        created_at: str | None = None,
        finished_at: str | None = None,
        #: Owner stamp for startup recovery -- see
        #: :func:`current_owner_fields` and :meth:`reclaim_interrupted`. Pass
        #: both or neither; a record with only one is treated the same as
        #: a record with no owner at all by ``reclaim_interrupted``.
        owner_pid: int | None = None,
        owner_boot_id: str | None = None,
        owner_start_time: float | None = None,
        #: Open passthrough for a caller's OWN fields this module has never
        #: heard of (a project id, a routing decision, anything a future
        #: caller wants to attach) without forking this method's signature
        #: for every new need. Checked against ``_RESERVED_FIELDS`` so a
        #: caller cannot accidentally clobber ``job_id``/``kind``/
        #: ``objective``/``status`` through it.
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write one complete job record, omitting optional fields set to ``None``.

        One construction path matters here: the promoted implementation used
        both a helper and a hand-built consultant dict. Keeping every writer
        on this method prevents their durable record shapes from drifting.

        Deliberately NOT a read-modify-write merge onto whatever is
        currently stored: every call rebuilds the record from exactly the
        fields it passes, the same contract this method has always had.
        A field that must survive from one write to the next (``created_at``,
        most notably) is the CALLER's job to carry forward and re-pass, not
        something this method infers by peeking at the old record -- that
        keeps this method's behaviour obvious from its own call site alone.
        """
        if extra:
            collision = _RESERVED_FIELDS & extra.keys()
            if collision:
                raise ValueError(f"extra must not override reserved job fields: {sorted(collision)}")
        record: dict[str, Any] = {
            "job_id": job_id,
            "kind": tool_name,
            "objective": objective,
            "status": status,
        }
        optional: dict[str, Any] = {
            "plan_id": plan_id,
            "task_index": task_index,
            "plan_task_text": plan_task_text,
            "result": result,
            "error": error,
            "engine": engine,
            "model": model,
            "effort": effort,
            "execution_started": execution_started,
            "cost_usd": cost_usd,
            "cost_unknown": cost_unknown,
            "created_at": created_at,
            "finished_at": finished_at,
            "owner_pid": owner_pid,
            "owner_boot_id": owner_boot_id,
            "owner_start_time": owner_start_time,
        }
        for name, value in optional.items():
            if value is not None:
                record[name] = value
        if extra:
            record.update(extra)
        self._store.write(self._key(job_id), record)

    def begin_execution(self, job_id: str, *, extra: dict[str, Any] | None = None) -> bool:
        """CAS a queued running record to started, preserving unknown fields.

        False means recovery or another worker won. Never starts a missing,
        terminal, or already-started record. Legacy missing markers are allowed.
        """
        if extra and _RESERVED_FIELDS & extra.keys():
            raise ValueError("execution metadata cannot replace job identity or status")
        key = self._key(job_id)
        while True:
            current = self._store.read(key)
            if not isinstance(current, dict) or current.get("status") != "running":
                return False
            if current.get("execution_started") is True:
                return False
            if self._store.compare_and_swap(key, current, {**current, **(extra or {}), "execution_started": True}):
                return True

    def update(
        self, job_id: str, changes: dict[str, Any], *, only_active: bool = True, expected_started: bool | None = None
    ) -> bool:
        """CAS a partial update, retaining fields the caller does not own.

        ``expected_started`` optionally protects setup cleanup from overwriting
        another worker that crossed the execution boundary in the meantime.
        """
        if {"job_id", "kind", "objective"} & changes.keys():
            raise ValueError("update cannot replace job identity")
        key = self._key(job_id)
        while True:
            current = self._store.read(key)
            if not isinstance(current, dict):
                return False
            if only_active and current.get("status") not in ("running", "awaiting_approval"):
                return False
            if expected_started is not None and (current.get("execution_started") is True) != expected_started:
                return False
            if self._store.compare_and_swap(key, current, {**current, **changes}):
                return True

    def reclaim_interrupted(self, *, reclaim_unowned: bool = False, stale_after: float = 900.0) -> list[str]:
        """Mark jobs owned by a dead local process as interrupted.

        Pass ``reclaim_unowned=True`` after confirming that any legacy records
        without a process stamp have no live workers. CAS protects a terminal
        result that arrives during the scan.
        """
        reclaimed: list[str] = []
        for key, raw in self._store.items(prefix=self._prefix):
            if not isinstance(raw, dict) or raw.get("status") not in ("running", "awaiting_approval"):
                continue
            alive = owner_is_alive(raw)
            if alive is True:
                continue
            if alive is None and not reclaim_unowned:
                age = record_age(raw)
                if age is None:
                    entry = self._store.read_entry(key)
                    age = time.time() - entry.written_at if entry is not None else None
                if age is None or age < stale_after:
                    continue
                _logger.warning("Reclaiming stale unowned delegation job %r (age=%r seconds)", raw.get("job_id"), age)
            if self._store.compare_and_swap(key, raw, {**raw, "status": "interrupted"}):
                reclaimed.append(raw.get("job_id", key))
        return reclaimed

    def find(self, job_id: str) -> dict[str, Any] | None:
        """Find a job by full id or the short prefix shown by ``check_jobs``.

        An 8-character uuid4 prefix collision needs thousands of jobs in one
        Store, so first match wins deliberately rather than adding ambiguity
        handling to the model-facing lookup path.
        """
        exact = self._store.read(self._key(job_id))
        if isinstance(exact, dict):
            return exact
        for _key, raw in self._store.items(prefix=self._prefix):
            if isinstance(raw, dict) and str(raw.get("job_id", "")).startswith(job_id):
                return raw
        return None

    def checker_tool(self, doc: str = "Check background delegation job statuses and result previews.") -> Tool:
        """Build the compact, many-job polling tool."""

        def check_jobs() -> str:
            jobs = [raw for _key, raw in self._store.items(prefix=self._prefix) if isinstance(raw, dict)]
            if not jobs:
                return "no jobs yet"
            jobs.sort(key=lambda job: job.get("job_id", ""))
            lines: list[str] = []
            for job in jobs:
                short_id = str(job.get("job_id", "?"))[:8]
                kind = job.get("kind", "?")
                status = job.get("status", "?")
                objective = (job.get("objective") or "")[:100]
                line = f"- {short_id} ({kind}) [{status}] {objective}"
                if job.get("plan_id") is not None:
                    line += (
                        f"\n  plan: {job['plan_id']} task {job.get('task_index', '?')}: "
                        f"{str(job.get('plan_task_text', ''))[:100]}"
                    )
                if status == "done" and job.get("result"):
                    line += f"\n  result: {str(job['result'])[:200]}"
                elif status in ("failed", "interrupted", "denied") and job.get("error"):
                    line += f"\n  error: {str(job['error'])[:200]}"
                lines.append(line)
            return "\n".join(lines)

        check_jobs.__doc__ = doc
        return Tool.wrap(check_jobs, name="check_jobs")

    def result_tool(self, doc: str = "Get one background job's complete, untruncated result.") -> Tool:
        """Build the untruncated escape hatch for compact checker previews."""

        def get_job_result(job_id: str) -> str:
            job = self.find(job_id)
            if job is None:
                return f"no job found matching {job_id!r}"
            lines = [
                f"job_id: {job.get('job_id', '?')}",
                f"kind: {job.get('kind', '?')}",
                f"status: {job.get('status', '?')}",
            ]
            if job.get("plan_id") is not None:
                lines.extend(
                    [
                        f"plan_id: {job['plan_id']}",
                        f"task_index: {job.get('task_index', '?')}",
                        f"plan_task_text: {job.get('plan_task_text', '')}",
                    ]
                )
            for field in ("engine", "model", "effort", "created_at", "finished_at"):
                if job.get(field) is not None:
                    lines.append(f"{field}: {job[field]}")
            if job.get("cost_usd") is not None:
                lines.append(f"cost_usd: {job['cost_usd']}")
            elif job.get("cost_unknown"):
                lines.append("cost_usd: unknown")
            objective = job.get("objective")
            if objective:
                lines.append(f"objective:\n{objective}")
            if job.get("result"):
                lines.append(f"result:\n{job['result']}")
            if job.get("error"):
                lines.append(f"error:\n{job['error']}")
            return "\n\n".join(lines)

        get_job_result.__doc__ = doc
        return Tool.wrap(get_job_result, name="get_job_result")


def _job_owner_is_alive(record: dict[str, Any]) -> bool | None:
    """Deprecated compatibility alias for :func:`owner_is_alive`."""
    return owner_is_alive(record)
