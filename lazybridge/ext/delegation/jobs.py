"""Durable job records and workspace-scoped session keys for delegation.

Promoted from LazyCEO's background delegation plumbing into an extension
under the core-vs-ext policy (``docs/guides/core-vs-ext.md``). The record
shape intentionally remains compatible with the source project while the
default Store namespaces are neutral; callers migrating an existing Store
can pass the old prefixes explicitly.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from lazybridge import Store, Tool

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


#: This process's own identity for job-ownership bookkeeping (see
#: :func:`_job_owner_is_alive`) -- a fresh uuid every time this module is
#: imported into a running process, paired with its pid. Two different
#: processes can share a pid over time (any OS recycles them), so pid ALONE
#: never proves "this is the same process that wrote this record" -- only
#: the pair does, and only for this exact process's own claim of
#: self-identity ("is this owner me"). Liveness of a DIFFERENT (or
#: unidentifiable) owner is checked against the real process table instead
#: (:func:`_pid_is_running`), since there is no registry of other
#: processes' boot ids to compare against. Promoted from LazyCEO's
#: ``simple.agent`` module, which found this the hard way: a bare pid
#: reused by an unrelated process made ``reclaim_interrupted`` think a dead
#: job's owner was still alive.
_PROCESS_BOOT_ID = str(uuid.uuid4())


def current_owner_fields() -> dict[str, Any]:
    """The owner stamp a long-running job's record should carry from the
    moment its worker actually starts -- ``os.getpid()`` paired with this
    process's own :data:`_PROCESS_BOOT_ID`. Pass the result as
    ``owner_pid=``/``owner_boot_id=`` to :meth:`JobRegistry.write`.
    Deliberately NOT stamped automatically by every write: a caller that
    never cares about cross-restart liveness (most tests, a short-lived
    CLI) should not have to see these fields in every record, and a record
    written for bookkeeping only (e.g. a denial) was never "owned" by a
    worker in the first place.
    """
    return {"owner_pid": os.getpid(), "owner_boot_id": _PROCESS_BOOT_ID}


def _pid_is_running(pid: int, *, timeout: float = 5.0) -> bool:
    """Best-effort liveness check for a FOREIGN owner pid.

    Windows: shells out to ``tasklist`` (no ``psutil`` dependency). Any
    other OS: ``os.kill(pid, 0)``, which sends no signal and only probes
    whether the pid exists and is reachable.

    Assumes ALIVE on any failure, timeout, or unexpected output: wrongly
    treating a still-running writer as dead risks a second worker starting
    on top of it (a real repository/workspace side effect duplicated),
    which is worse than conservatively leaving a job alone for one more
    sweep -- the same "unknown is never free capacity" direction a quota
    brake would take for telemetry it cannot read. Promoted from LazyCEO's
    ``simple.agent._pid_is_running``.
    """
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception:
            return True
        if result.returncode != 0:
            return True
        output = result.stdout.strip()
        if not output or output.upper().startswith("INFO:"):
            return False
        return str(pid) in output
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, but owned by another user -- still alive.
        return True
    except Exception:
        return True
    return True


def _job_owner_is_alive(raw: dict[str, Any]) -> bool:
    """Whether the process that last claimed ownership of this job record is
    still running.

    A record with no owner stamp at all (never written with
    ``owner_pid``/``owner_boot_id``, e.g. everything written before a
    caller opted into :func:`current_owner_fields`) answers False -- "not
    confirmed alive" -- which is exactly what :meth:`JobRegistry.
    reclaim_interrupted` assumed before this field existed. Anything whose
    ``owner_pid`` differs from this process's own is checked against the
    real table (:func:`_pid_is_running`).

    A record whose ``owner_pid`` equals THIS process's own is handled as
    its own case, not folded into the ``_pid_is_running`` branch: only a
    MATCHING ``owner_boot_id`` means it is actually asking about itself
    (alive by definition -- no process-table lookup can answer that better
    than the process itself). A matching pid with a DIFFERENT boot id means
    pid reuse: an earlier, now-dead process happened to be handed the exact
    pid the OS later gave this one. Falling through to ``_pid_is_running``
    there would trivially see THIS process (it has that pid, it exists) and
    wrongly report the old, dead owner as alive, leaving a genuinely
    orphaned job stuck ``"running"`` forever after a restart that recycled
    its old owner's pid. Found by Codex review before this ever shipped.
    """
    owner_pid = raw.get("owner_pid")
    owner_boot_id = raw.get("owner_boot_id")
    if not isinstance(owner_pid, int) or not isinstance(owner_boot_id, str):
        return False
    if owner_pid == os.getpid():
        return owner_boot_id == _PROCESS_BOOT_ID
    return _pid_is_running(owner_pid)


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
        #: Owner stamp for cross-restart liveness -- see
        #: :func:`current_owner_fields`/:func:`_job_owner_is_alive`. Pass
        #: both or neither; a record with only one is treated the same as
        #: a record with no owner at all by ``reclaim_interrupted``.
        owner_pid: int | None = None,
        owner_boot_id: str | None = None,
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
        }
        for name, value in optional.items():
            if value is not None:
                record[name] = value
        if extra:
            record.update(extra)
        self._store.write(self._key(job_id), record)

    def reclaim_interrupted(self) -> list[str]:
        """Mark orphaned in-progress jobs ``interrupted`` -- but only the
        ones whose OWNER is actually dead.

        Meant to be called at process startup. A background asyncio Task
        does not survive across processes, so an old ``running`` or
        ``awaiting_approval`` record left by a process that has since
        exited has no coroutine (or live approval wait) left to resume.
        ``interrupted`` is more accurate than ``failed`` and avoids
        silently presenting the job as active.

        Ownership-aware: a record stamped with ``owner_pid``/
        ``owner_boot_id`` (see :func:`current_owner_fields`) is left alone
        when :func:`_job_owner_is_alive` confirms that process is still
        running -- more than one process can share a ``Store`` and be
        concurrently alive at once (the exact case the un-ownership-aware
        version of this method could not tell apart from a genuinely dead
        writer), and this is what makes that safe. A record with no owner
        stamp at all answers "not confirmed alive" the same way the
        un-ownership-aware version always treated every in-progress record,
        so a caller that never adopts :func:`current_owner_fields` keeps
        the exact old behaviour. Promoted from LazyCEO's
        ``simple.agent``'s owner-aware startup recovery.

        Compare-and-swap is essential regardless: a genuine old-process
        completion (or an owner dying between the read above and this
        write) may land between this scan and the update. A lost CAS skips
        that job so its real ``done``/``failed`` result is never
        clobbered.
        """
        reclaimed: list[str] = []
        for key, raw in self._store.items(prefix=self._prefix):
            if (
                isinstance(raw, dict)
                and raw.get("status") in ("running", "awaiting_approval")
                and not _job_owner_is_alive(raw)
                and self._store.compare_and_swap(key, raw, {**raw, "status": "interrupted"})
            ):
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
