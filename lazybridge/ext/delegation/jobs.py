"""Durable job records and workspace-scoped session keys for delegation.

Promoted from LazyCEO's background delegation plumbing into an extension
under the core-vs-ext policy (``docs/guides/core-vs-ext.md``). The record
shape intentionally remains compatible with the source project while the
default Store namespaces are neutral; callers migrating an existing Store
can pass the old prefixes explicitly.
"""

from __future__ import annotations

import hashlib
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
    ) -> None:
        """Write one complete job record, omitting optional fields set to ``None``.

        One construction path matters here: the promoted implementation used
        both a helper and a hand-built consultant dict. Keeping every writer
        on this method prevents their durable record shapes from drifting.
        """
        record: dict[str, Any] = {
            "job_id": job_id,
            "kind": tool_name,
            "objective": objective,
            "status": status,
        }
        if plan_id is not None:
            record["plan_id"] = plan_id
        if task_index is not None:
            record["task_index"] = task_index
        if plan_task_text is not None:
            record["plan_task_text"] = plan_task_text
        if result is not None:
            record["result"] = result
        if error is not None:
            record["error"] = error
        self._store.write(self._key(job_id), record)

    def reclaim_interrupted(self) -> list[str]:
        """Mark process-orphaned in-progress jobs ``interrupted``.

        Meant to be called ONCE, at process startup. A background asyncio
        Task does not survive across processes, so an old ``running`` or
        ``awaiting_approval`` record has no coroutine (or live approval
        wait) left to resume. ``interrupted`` is more accurate than
        ``failed`` and avoids silently presenting the job as active.

        Compare-and-swap is essential: a genuine old-process completion may
        land between this scan and the update. A lost CAS skips that job so
        its real ``done``/``failed`` result is never clobbered.

        Known accepted constraint: a job record carries no process/run
        identity or lease, only a status. If more than one process shares
        the same ``Store`` and is concurrently ALIVE at once (not the
        "old process died, new process starts up" case this method is
        for), this can't tell a genuinely still-running job in another
        live process from an orphaned one, and would wrongly interrupt
        it. The CAS only protects against a job finishing between this
        scan and the write; it provides no cross-process liveness check.
        This mirrors the promoted source's own single-owning-process
        assumption (its own docstring reasons purely about "a background
        Task does not survive across processes," never about concurrent
        processes). Real multi-process safety would need a lease/heartbeat
        per job, a bigger addition than this extraction takes on; revisit
        only if a real concurrent-process deployment needs it. Found by
        Codex review before this ever shipped.
        """
        reclaimed: list[str] = []
        for key, raw in self._store.items(prefix=self._prefix):
            if (
                isinstance(raw, dict)
                and raw.get("status") in ("running", "awaiting_approval")
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
