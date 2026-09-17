"""Durable blackboard — a to-do list that outlives the process.

The sibling :mod:`lazybridge.ext.planners.blackboard` keeps its plan in a
closure dict and **resets it on every invocation**, which is right for a
one-shot planner and wrong for an always-on agent: such an agent wakes up,
does one thing, and must still know where it is after a crash or a restart.

This module keeps the same three-verb feel (plan → work → tick) but stores
the plan in a :class:`~lazybridge.Store`, and adds the two things a resumable
worker actually needs:

* **claiming** — ``claim_next`` hands out exactly one task, atomically, so two
  workers on the same plan never take the same item;
* **leases and attempts** — a claim that is never closed (the worker died)
  expires and becomes claimable again, and a task that keeps killing its
  worker is parked as ``failed`` after ``max_attempts`` instead of looping
  forever.

Nothing here is agent-specific: :class:`DurableBlackboard` is usable on its
own, and :func:`durable_blackboard_agent` wraps it as an ``Agent`` whose
tools are the blackboard verbs.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from lazybridge import Agent, LLMEngine, Store, Tool

TaskStatus = Literal["todo", "claimed", "done", "failed", "cancelled"]

#: Bumped when the persisted document shape changes incompatibly.
BLACKBOARD_VERSION = 1

DURABLE_BLACKBOARD_GUIDANCE = """\
# How to work

You keep a durable plan that survives restarts. It is the only place your
progress is recorded — your own memory of this conversation is not.

Tools:

- ``get_plan()``                       — read the plan and see what is left.
- ``set_plan(reasoning, tasks)``       — create the plan (only when there is none).
- ``add_tasks(tasks)``                 — append newly-discovered tasks to the plan.
- ``cancel_task(task_index, expected_text, reason)`` — drop a todo task that is no longer needed.
- ``set_task_schedule(task_index, expected_text, planned_start_at, due_at, reason)`` — schedule an open task.
- ``claim_next()``                     — take the next task; returns its index and text.
- ``claim_task(task_index, expected_text)`` — take a SPECIFIC task instead of "next", when you already know which one you need (expected_text must match get_plan()'s current text for that index exactly).
- ``mark_done(task_index, summary)``   — close a task with a 1-3 sentence result.
- ``mark_failed(task_index, error)``   — give a task back after a real failure.

## Workflow

1. Always start with ``get_plan()``.
2. If there is no plan, call ``set_plan`` with 3-6 coarse, self-contained
   tasks in execution order, then stop and report the plan.
3. If there is a plan and you have no specific task in mind, call
   ``claim_next()``. If you already know exactly which task you need (its
   index and current text from ``get_plan()``) — for example to reach one
   sitting behind others you deliberately are not doing yet — call
   ``claim_task(task_index, expected_text)`` instead of claiming and closing
   every earlier task just to advance past them. Either way: once a task is
   claimed, do **that one task only**, then ``mark_done`` it and report what
   you did. Do not claim a second task in the same run.
4. If ``claim_next()``/``claim_task()`` says the plan is complete (or that
   nothing is claimable), report the final result.
5. If the work genuinely fails, call ``mark_failed`` with the reason —
   the task returns to the queue and is retried on a later run.

Never invent progress: a task counts as done only once ``mark_done`` returns.
"""


def _fresh_task(text: str) -> dict[str, Any]:
    """A brand-new ``todo`` task record -- the one place the field list is
    spelled out, so ``set_plan`` and ``add_tasks`` can never drift apart on
    what a task looks like at creation."""
    return {
        "text": str(text),
        "status": "todo",
        "result": "",
        "error": "",
        "cancel_reason": "",
        "attempts": 0,
        "owner": None,
        "claimed_at": None,
        "planned_start_at": None,
        "due_at": None,
        #: ``time.time()`` (same convention as ``claimed_at``) the moment
        #: this task reaches a TERMINAL state: ``done``, an exhausted
        #: ``failed``, or ``cancelled``. None while the task is open --
        #: including a ``mark_failed`` that sends it back to ``todo`` for a
        #: retry, which is not a close. Without this, a consumer reading a
        #: closed task's text has no way to place it in time at all; a real
        #: incident (a report showing days-old closures under today's date)
        #: is what this field exists to close off.
        "completed_at": None,
    }


@dataclass(frozen=True)
class BlackboardSnapshot:
    """Read-only view of a plan, for callers that want data instead of text."""

    plan_id: str
    reasoning: str
    tasks: list[dict[str, Any]]
    schedule_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.tasks) and all(t["status"] in ("done", "failed", "cancelled") for t in self.tasks)

    @property
    def open_tasks(self) -> list[int]:
        return [i for i, t in enumerate(self.tasks) if t["status"] in ("todo", "claimed")]


class DurableBlackboard:
    """A plan of tasks persisted in a :class:`~lazybridge.Store`.

    Every mutation is a compare-and-swap on the whole document, so two
    workers sharing a ``Store`` serialise instead of overwriting each other.
    """

    def __init__(
        self,
        store: Store,
        plan_id: str,
        *,
        lease_seconds: float = 900.0,
        max_attempts: int = 3,
        key_prefix: str = "blackboard:",
        cas_retries: int = 8,
    ) -> None:
        if not plan_id:
            raise ValueError("plan_id must be a non-empty string")
        if lease_seconds <= 0:
            raise ValueError(f"lease_seconds must be > 0, got {lease_seconds!r}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}")
        self.store = store
        self.plan_id = plan_id
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.key = f"{key_prefix}{plan_id}"
        self._cas_retries = cas_retries

    # -- persistence ----------------------------------------------------

    def _read(self) -> dict[str, Any] | None:
        doc = self.store.read(self.key)
        return doc if isinstance(doc, dict) else None

    def _mutate(self, apply: Any) -> Any:
        """Read-modify-write under compare-and-swap.

        ``apply(doc)`` returns ``(new_doc, result)``; returning ``None`` for
        ``new_doc`` means "no write needed" and short-circuits. A lost race
        re-reads and retries rather than clobbering the other writer.
        """
        for _ in range(self._cas_retries):
            current = self._read()
            new_doc, result = apply(current)
            if new_doc is None:
                return result
            new_doc["updated_at"] = time.time()
            if self.store.compare_and_swap(self.key, current, new_doc):
                return result
        raise RuntimeError(
            f"blackboard {self.plan_id!r}: gave up after {self._cas_retries} lost races — "
            "another worker is writing continuously"
        )

    # -- plan lifecycle -------------------------------------------------

    def set_plan(self, reasoning: str, tasks: list[str], *, replace: bool = False) -> str:
        """Create the plan. Refuses to discard an unfinished one unless ``replace``."""
        if not reasoning.strip() or not tasks:
            return "REJECTED: reasoning and a non-empty tasks list are both required."

        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
            if doc is not None and not replace:
                snapshot = self._snapshot_of(doc)
                if snapshot.open_tasks:
                    return None, (
                        "REJECTED: a plan is already in progress — call get_plan() and "
                        "continue it. Pass replace=True only to deliberately abandon it.\n" + self._render(doc)
                    )
            fresh = {
                "version": BLACKBOARD_VERSION,
                "plan_id": self.plan_id,
                "reasoning": reasoning.strip(),
                "created_at": time.time(),
                "tasks": [_fresh_task(t) for t in tasks],
                "schedule_events": [],
            }
            return fresh, self._render(fresh)

        return str(self._mutate(apply))

    def add_tasks(self, tasks: list[str]) -> str:
        """Append new tasks to an in-progress plan without touching any
        existing task's index or status -- the low-risk half of "revise the
        plan without discarding it": pure append never shifts an index a
        worker may already be holding from ``claim_next``, unlike deletion
        or reordering would.
        """
        clean = [str(t).strip() for t in tasks if str(t).strip()]
        if not clean:
            return "REJECTED: tasks must contain at least one non-empty item."

        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
            if doc is None or not doc.get("tasks"):
                return None, "REJECTED: no plan yet; call set_plan first."
            new_doc = {**doc, "tasks": [*doc["tasks"], *(_fresh_task(t) for t in clean)]}
            return new_doc, self._render(new_doc)

        return str(self._mutate(apply))

    def cancel_task(self, task_index: int, expected_text: str, reason: str) -> str:
        """Permanently drop a not-yet-claimed task, without needing to claim
        it first (unlike ``mark_done``/``mark_failed``, which require an
        active claim).

        ``expected_text`` must match the task's CURRENT text exactly. This
        is the guard against acting on a stale rendering: an LLM that still
        remembers an earlier ``get_plan()`` (before a concurrent
        ``add_tasks``/``claim_next`` changed what index N refers to) gets a
        clear rejection instead of silently cancelling the wrong task. Only
        a ``todo`` task may be cancelled -- a claimed task belongs to its
        worker until that worker closes it; cancel must never pre-empt it.
        """
        if not reason.strip():
            return "REJECTED: a reason is required."

        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
            if doc is None or not doc.get("tasks"):
                return None, "REJECTED: no plan set; call set_plan first."
            tasks = [dict(t) for t in doc["tasks"]]
            if not 0 <= task_index < len(tasks):
                return None, f"REJECTED: task_index out of range (valid: 0..{len(tasks) - 1})."
            task = tasks[task_index]
            if task["text"] != expected_text:
                return None, (
                    f"REJECTED: task {task_index}'s current text does not match expected_text -- "
                    "call get_plan() to see the current state before cancelling."
                )
            if task.get("status") != "todo":
                return None, (
                    f"REJECTED: task {task_index} is {task.get('status')}, not todo -- "
                    "only an unclaimed task can be cancelled."
                )
            task.update(status="cancelled", cancel_reason=reason.strip(), completed_at=time.time())
            new_doc = {**doc, "tasks": tasks}
            return new_doc, self._render(new_doc)

        return str(self._mutate(apply))

    def set_task_schedule(
        self,
        task_index: int,
        expected_text: str,
        *,
        planned_start_at: float | None = None,
        due_at: float | None = None,
        reason: str,
    ) -> str:
        """Set or clear a non-terminal task's schedule and record the edit.

        The task update and its append-only audit event are part of the same
        whole-document compare-and-swap. A concurrent close therefore either
        preserves this schedule or wins first and causes this edit to be
        rejected after the CAS retry re-reads the terminal task.
        """
        for name, value in (("planned_start_at", planned_start_at), ("due_at", due_at)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{name} must be a finite number or None, got {value!r}")
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if planned_start_at is not None and due_at is not None and planned_start_at > due_at:
            raise ValueError("planned_start_at must be <= due_at")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")

        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
            if doc is None or not doc.get("tasks"):
                return None, "REJECTED: no plan set; call set_plan first."
            tasks = [dict(t) for t in doc["tasks"]]
            if not 0 <= task_index < len(tasks):
                return None, f"REJECTED: task_index out of range (valid: 0..{len(tasks) - 1})."
            task = tasks[task_index]
            if task["text"] != expected_text:
                return None, (
                    f"REJECTED: task {task_index}'s current text does not match expected_text -- "
                    "call get_plan() to see the current state before scheduling."
                )
            if task.get("status") not in ("todo", "claimed"):
                return None, (
                    f"REJECTED: task {task_index} is {task.get('status')}, not open -- "
                    "only a todo or claimed task can be scheduled."
                )

            event_at = time.time()
            task.update(planned_start_at=planned_start_at, due_at=due_at)
            event = {
                "task_index": task_index,
                "planned_start_at": planned_start_at,
                "due_at": due_at,
                "reason": reason.strip(),
                "at": event_at,
            }
            new_doc = {
                **doc,
                "tasks": tasks,
                "schedule_events": [*doc.get("schedule_events", []), event],
            }
            return new_doc, self._render(new_doc)

        return str(self._mutate(apply))

    def snapshot(self) -> BlackboardSnapshot:
        doc = self._read()
        return self._snapshot_of(doc)

    def _snapshot_of(self, doc: dict[str, Any] | None) -> BlackboardSnapshot:
        if doc is None:
            return BlackboardSnapshot(plan_id=self.plan_id, reasoning="", tasks=[])
        return BlackboardSnapshot(
            plan_id=str(doc.get("plan_id", self.plan_id)),
            reasoning=str(doc.get("reasoning", "")),
            tasks=list(doc.get("tasks", [])),
            schedule_events=list(doc.get("schedule_events", [])),
        )

    def render(self) -> str:
        """Human/LLM-readable state, including which task is next."""
        return self._render(self._read())

    def _render(self, doc: dict[str, Any] | None) -> str:
        if doc is None or not doc.get("tasks"):
            return "(no plan yet; call set_plan)"
        marks = {"todo": "[ ]", "claimed": "[~]", "done": "[x]", "failed": "[!]", "cancelled": "[-]"}
        lines = [f"plan: {doc.get('plan_id')}", f"reasoning: {doc.get('reasoning', '')}"]
        for i, task in enumerate(doc["tasks"]):
            row = f"  {i}. {marks.get(task['status'], '[?]')} {task['text']}"
            planned_start_at = task.get("planned_start_at")
            due_at = task.get("due_at")
            if planned_start_at is not None:
                row += f"\n       planned start: {planned_start_at}"
            if due_at is not None:
                row += f"\n       due: {due_at}"
                if task.get("status") in ("todo", "claimed") and due_at < time.time():
                    row += " OVERDUE"
            if task.get("result"):
                row += f"\n       → {task['result']}"
            if task.get("error"):
                row += f"\n       ! {task['error']} (attempts: {task.get('attempts', 0)})"
            if task.get("cancel_reason"):
                row += f"\n       (cancelled: {task['cancel_reason']})"
            lines.append(row)
        snapshot = self._snapshot_of(doc)
        if snapshot.complete:
            done = sum(1 for t in snapshot.tasks if t["status"] == "done")
            failed = sum(1 for t in snapshot.tasks if t["status"] == "failed")
            cancelled = sum(1 for t in snapshot.tasks if t["status"] == "cancelled")
            if failed or cancelled:
                lines.append(f"plan complete — {done} done, {failed} failed, {cancelled} cancelled")
            else:
                lines.append("plan complete — all tasks done")
        else:
            nxt = next((i for i, t in enumerate(snapshot.tasks) if t["status"] == "todo"), None)
            lines.append(f"next claimable: {nxt}" if nxt is not None else "no free task right now (all claimed)")
        return "\n".join(lines)

    # -- work -----------------------------------------------------------

    def claim_next(self, owner: str | None = None) -> tuple[int, str] | None:
        """Take the next task atomically, or ``None`` when there is nothing to take.

        A task still ``claimed`` past its lease is treated as abandoned — the
        worker holding it died — and is handed out again, its attempt already
        counted.
        """
        holder = owner or uuid.uuid4().hex

        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, tuple[int, str] | None]:
            if doc is None or not doc.get("tasks"):
                return None, None
            now = time.time()
            tasks = [dict(t) for t in doc["tasks"]]
            # Earliest ELIGIBLE index across todo *and* expired-claimed, not
            # todo-always-first: preferring todo unconditionally means a
            # plan that keeps growing via add_tasks() can starve reclaiming
            # an abandoned worker's task forever, since a fresh todo item is
            # always available before the scan ever reaches the expired
            # claim sitting earlier in the list.
            todo_index = next((i for i, t in enumerate(tasks) if t["status"] == "todo"), None)
            expired_index = next(
                (
                    i
                    for i, t in enumerate(tasks)
                    if t["status"] == "claimed"
                    and t.get("claimed_at") is not None
                    and now - float(t["claimed_at"]) > self.lease_seconds
                ),
                None,
            )
            candidates = [i for i in (todo_index, expired_index) if i is not None]
            index = min(candidates) if candidates else None
            if index is None:
                return None, None
            task = tasks[index]
            if task.get("attempts", 0) >= self.max_attempts:
                # Out of retries: park it so the plan can finish instead of
                # handing the same poison task out forever.
                task.update(status="failed", owner=None, claimed_at=None, completed_at=time.time())
                task["error"] = task.get("error") or f"exhausted {self.max_attempts} attempts"
                new_doc = {**doc, "tasks": tasks}
                return new_doc, None
            task.update(
                status="claimed",
                owner=holder,
                claimed_at=now,
                attempts=task.get("attempts", 0) + 1,
                # A fresh claim is a fresh right to renew once. Without this,
                # a task renewed by a prior holder that later became
                # reclaimable (lease lapsed, or reset to todo) would keep
                # renewed=True across the new holder's own claim -- so THAT
                # holder's first legitimate claim_task(renew=True) would be
                # refused as if it had already used its one renewal, which
                # it never did. claim_project_task actually reaches this
                # path (not just claim_task's own fresh-claim branch): it
                # calls claim_next() as its default/fallback route, on the
                # same project board delegate_project_task later renews.
                # Found by independent review while verifying an unrelated
                # task on this same board.
                renewed=False,
            )
            return {**doc, "tasks": tasks}, (index, str(task["text"]))

        # Parking a poison task is itself a mutation that hands back nothing,
        # so retry while progress is still possible. Bounded by the task count:
        # each pass either claims something or parks one task for good.
        for _ in range(len(self.snapshot().tasks) + 1):
            claimed = self._mutate(apply)
            if claimed is not None:
                return claimed  # type: ignore[no-any-return]
            if not any(t["status"] == "todo" for t in self.snapshot().tasks):
                return None
        return None

    def claim_task(
        self, task_index: int, expected_text: str, *, owner: str | None = None, renew: bool = False
    ) -> tuple[int, str] | str:
        """Take a SPECIFIC task instead of whichever is earliest-eligible --
        for a caller that already knows which task it wants and does not
        want to claim (and immediately close) every earlier todo task just
        to advance ``claim_next``'s scan past them.

        Found live: an agent whose plan had tasks like "[1] optional
        cleanup, only if the user explicitly asks" sitting before the task
        it actually needed to work on right now had no way to reach that
        later task without claiming AND immediately closing [1] first --
        polluting the plan's history with noisy, fake-looking "no action
        taken, closing only to unblock the queue" result summaries that
        were not real completions.

        This is an alternate SELECTION, not a different claiming mechanism
        -- ``claim_next`` remains the only way an unattended worker picks up
        arbitrary work, and its own scan/lease/attempts behaviour is
        unchanged by this method existing. A targeted task still ends up
        ``claimed``, still counts an attempt, still parks as ``failed`` if
        attempts are already exhausted, and a task actively claimed by
        another (unexpired) worker is refused here exactly as ``claim_next``
        would never hand it out to a second worker.

        ``expected_text`` is required and must match the task's CURRENT
        text -- the same stale-index guard ``cancel_task`` already uses: an
        index the caller remembers from an earlier ``get_plan()`` may not
        refer to the same task anymore after a concurrent
        ``add_tasks()``/``claim_next()`` elsewhere changed what that index
        holds.

        Returns ``(index, text)`` on success, the same shape ``claim_next``
        returns, or a ``"REJECTED: ..."`` string explaining why not --
        distinguish the two with ``isinstance(result, str)``, the same
        convention ``cancel_task``/``mark_done``/``mark_failed`` already use
        for their own rejections.
        """
        holder = owner or uuid.uuid4().hex

        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, tuple[int, str] | str]:
            if doc is None or not doc.get("tasks"):
                return None, "REJECTED: no plan set; call set_plan first."
            tasks = [dict(t) for t in doc["tasks"]]
            if not 0 <= task_index < len(tasks):
                return None, f"REJECTED: task_index out of range (valid: 0..{len(tasks) - 1})."
            task = tasks[task_index]
            if task["text"] != expected_text:
                return None, (
                    f"REJECTED: task {task_index}'s current text does not match expected_text -- "
                    "call get_plan() to see the current state before claiming."
                )
            now = time.time()
            # A worker re-entering a task it already holds is not a second
            # worker. The check below used to compare only the lease clock,
            # never the owner, so the holder was refused its own task and
            # told another worker had it -- with no other worker in the
            # system. LazyCEO's contracted-task flow does exactly this
            # re-entry (claim_project_task, then delegate_project_task
            # claims the same index to attach the job), so the flow could
            # not complete: seen live on project lazyceostudio, where a
            # healthy task was driven to 'failed' at attempts=3 on purpose,
            # to escape a competitor that did not exist.
            claimed_at = task.get("claimed_at")
            expired = claimed_at is not None and now - float(claimed_at) > self.lease_seconds
            renewing = (
                task.get("status") == "claimed"
                # owner=None means "I am not identifying myself" -- two
                # anonymous callers are not the same caller, and must not
                # be able to walk into each other's leases.
                and owner is not None
                and task.get("owner") == holder
                # Opt-in, so the default stays "refuse". An owner here is
                # PROCESS-level, not per-worker: delegate_plan_tasks claims
                # every item of a batch under one owner string, so owner
                # equality alone cannot tell "the holder re-entering to
                # attach a job" from "a second, genuinely new delegation
                # for a task that already has one running". Inferring
                # renewal from owner equality would let the same batch
                # start two writing jobs on one task and break the
                # single-worker guarantee this claim exists to give. Only a
                # caller that KNOWS it is re-entering passes renew=True.
                # Found by Codex review on PR #169.
                and renew
                # Only while the lease is still alive. A holder whose lease
                # ran out did NOT finish in time, and that is exactly what
                # an attempt counts. Without this, a stable owner identity
                # -- which every LazyCEO agent has, the same string across
                # restarts -- renews forever after each stall-kill, attempts
                # never grows, and a genuinely poisoned task is never
                # parked. The watchdog restarts this agent routinely, so
                # that is the normal path, not an edge case. Found by Codex
                # review.
                and not expired
                # And only ONCE. A renewal exists to attach a worker to
                # a task; a second one attaches a second worker to the
                # same checkout. Checking that OUTSIDE this function
                # cannot work -- scan, claim and record are then three
                # separate steps, and two concurrent callers both read
                # "nothing running" before either writes. The claim is
                # the mutual exclusion primitive, so the exclusion has
                # to live inside its compare-and-swap, not beside it.
                # Cleared by the fresh-claim path below -- which is what
                # a lapsed lease turns into -- so a worker that dies
                # holding a renewal releases it the same way it releases
                # the claim, with no separate reaper to get stuck.
                # Found by Codex review on PR #45.
                and not task.get("renewed")
            )
            if task.get("status") == "claimed" and not renewing:
                if not expired:
                    return None, (
                        f"REJECTED: task {task_index} is already claimed by another worker "
                        "and its lease has not expired yet -- wait, or work on something else."
                    )
            elif task.get("status") not in ("todo", "claimed"):
                return None, f"REJECTED: task {task_index} is {task.get('status')}, not claimable."
            if renewing:
                # Extend the lease and hand back the same task. Deliberately
                # NOT counted as an attempt: attempts is the poison-task
                # budget -- how many times this work has been tried and not
                # finished -- and a holder stepping back into a task it
                # never left has not tried it again. Counting it would walk
                # a healthy task to 'failed' in three ordinary steps of the
                # intended flow, which is what happened.
                task.update(claimed_at=now, renewed=True)
                new_doc = {**doc, "tasks": tasks}
                return new_doc, (task_index, str(task["text"]))
            if task.get("attempts", 0) >= self.max_attempts:
                # Same poison-task handling as claim_next: park it so the
                # plan can finish instead of handing it out again.
                task.update(status="failed", owner=None, claimed_at=None, completed_at=time.time())
                task["error"] = task.get("error") or f"exhausted {self.max_attempts} attempts"
                new_doc = {**doc, "tasks": tasks}
                return new_doc, (
                    f"REJECTED: task {task_index} just exhausted its attempt budget and was parked as failed instead."
                )
            task.update(
                status="claimed",
                owner=holder,
                claimed_at=now,
                attempts=task.get("attempts", 0) + 1,
                # A fresh claim is a fresh right to renew once.
                renewed=False,
            )
            new_doc = {**doc, "tasks": tasks}
            return new_doc, (task_index, str(task["text"]))

        return self._mutate(apply)

    def mark_done(self, task_index: int, summary: str, *, owner: str | None = None) -> str:
        if not summary.strip():
            return "REJECTED: a 1-3 sentence summary is required."
        return self._close(task_index, owner=owner, status="done", text=summary.strip())

    def mark_failed(self, task_index: int, error: str, *, owner: str | None = None) -> str:
        if not error.strip():
            return "REJECTED: an error description is required."
        return self._close(task_index, owner=owner, status="todo", text=error.strip())

    def _close(self, task_index: int, *, owner: str | None, status: TaskStatus, text: str) -> str:
        def apply(doc: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
            if doc is None or not doc.get("tasks"):
                return None, "REJECTED: no plan set; call set_plan first."
            tasks = [dict(t) for t in doc["tasks"]]
            if not 0 <= task_index < len(tasks):
                return None, f"REJECTED: task_index out of range (valid: 0..{len(tasks) - 1})."
            task = tasks[task_index]
            # Closing requires an *active* claim. Accepting ``owner is None``
            # as "unowned, therefore anyone may close it" would let a worker
            # whose lease expired come back and overwrite the result of the
            # run that replaced it — the closing worker clears ``owner``, so
            # by then the task looks unowned again. It would also let a task
            # be ticked without ever being claimed.
            if task.get("status") != "claimed":
                return None, (
                    f"REJECTED: task {task_index} is {task.get('status')}, not claimed — "
                    "call claim_next() before closing a task."
                )
            if owner is not None and task.get("owner") != owner:
                return None, (
                    f"REJECTED: task {task_index} is held by another worker — "
                    "its lease was reassigned while you were working."
                )
            if status == "done":
                task.update(status="done", result=text, error="", owner=None, claimed_at=None, completed_at=time.time())
            else:
                exhausted = task.get("attempts", 0) >= self.max_attempts
                # Only an EXHAUSTED failure actually closes the task --
                # sent back to "todo" for a retry is still open, and must
                # not look completed to a consumer reading completed_at.
                task.update(
                    status="failed" if exhausted else "todo",
                    error=text,
                    owner=None,
                    claimed_at=None,
                    completed_at=time.time() if exhausted else None,
                )
            new_doc = {**doc, "tasks": tasks}
            return new_doc, self._render(new_doc)

        return str(self._mutate(apply))


def durable_blackboard_agent(
    agents: list[Agent],
    *,
    store: Store,
    plan_id: str,
    engine: Any | None = None,
    model: str = "claude-opus-4-7",
    system: str | None = None,
    name: str = "durable_blackboard",
    worker_id: str | None = None,
    lease_seconds: float = 900.0,
    max_attempts: int = 3,
    verbose: bool = False,
) -> Agent:
    """An agent whose to-do list lives in ``store`` and survives restarts.

    Unlike :func:`~lazybridge.ext.planners.make_blackboard_planner`, nothing
    is reset between runs: call the returned agent repeatedly (a scheduler, a
    loop, a LazyPulse tick) and it picks up where the plan left off.

    Args:
        agents: Sub-agents the planner may call. Unique ``.name`` required.
        store: Where the plan lives. Use ``Store(db=...)`` to survive restarts.
        plan_id: Stable identity of this plan — the resume handle.
        engine: Pre-built engine (e.g. ``ClaudeCodeEngine()``); defaults to
            ``LLMEngine(model)``. A pre-built engine carries its own system
            prompt, so pass ``DURABLE_BLACKBOARD_GUIDANCE`` (or your own
            equivalent) to it yourself — this factory will not reach into it.
        worker_id: Identity this run claims tasks under. Defaults to a fresh
            random id per factory call, which is what separate runs need: two
            planners built from the same ``name`` are different workers, and a
            run whose lease expired must not be able to close the claim of the
            run that replaced it. Pin it only if something outside owns the
            identity.
        lease_seconds: How long a claimed task stays claimed before another
            run may take it over.
        max_attempts: Attempts per task before it is parked as ``failed``.
    """
    if not agents:
        raise ValueError("agents list must not be empty")
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"agents must have unique names; got {names}")

    board = DurableBlackboard(store, plan_id, lease_seconds=lease_seconds, max_attempts=max_attempts)
    holder = worker_id or f"{name}-{uuid.uuid4().hex[:8]}"

    def set_plan(reasoning: str, tasks: list[str]) -> str:
        """Create the plan: 3-6 coarse tasks in execution order. Refused if one is already open."""
        return board.set_plan(reasoning, tasks)

    def get_plan() -> str:
        """Read the durable plan: what is done, what is claimed, what is next."""
        return board.render()

    def add_tasks(tasks: list[str]) -> str:
        """Append newly-discovered tasks to the plan without disturbing any existing task."""
        return board.add_tasks(tasks)

    def cancel_task(task_index: int, expected_text: str, reason: str) -> str:
        """Drop a not-yet-claimed task that turned out to be unnecessary. expected_text must match
        the task's current text exactly (call get_plan() first) -- a mismatch is refused."""
        return board.cancel_task(task_index, expected_text, reason)

    def set_task_schedule(
        task_index: int,
        expected_text: str,
        planned_start_at: float | None = None,
        due_at: float | None = None,
        reason: str = "",
    ) -> str:
        """Set or clear an open task's planned start/due timestamps and record why."""
        return board.set_task_schedule(
            task_index,
            expected_text,
            planned_start_at=planned_start_at,
            due_at=due_at,
            reason=reason,
        )

    def claim_next() -> str:
        """Take the next task to work on. Do only that task this run."""
        claimed = board.claim_next(owner=holder)
        if claimed is None:
            snapshot = board.snapshot()
            if not snapshot.tasks:
                return "no plan yet; call set_plan first"
            if snapshot.complete:
                return "plan complete — summarise the result for the user"
            return "no free task right now — another worker holds the remaining ones"
        index, text = claimed
        return f"claimed task {index}: {text}"

    def claim_task(task_index: int, expected_text: str) -> str:
        """Take a SPECIFIC task instead of whichever is next, when you
        already know which one you need. expected_text must match the
        task's CURRENT text exactly (call get_plan() first) -- a mismatch
        is refused, same guard as cancel_task."""
        result = board.claim_task(task_index, expected_text, owner=holder)
        if isinstance(result, str):
            return result
        index, text = result
        return f"claimed task {index}: {text}"

    def mark_done(task_index: int, result_summary: str) -> str:
        """Close a task with a 1-3 sentence summary of what was produced."""
        return board.mark_done(task_index, result_summary, owner=holder)

    def mark_failed(task_index: int, error: str) -> str:
        """Give a task back after a real failure; it is retried on a later run."""
        return board.mark_failed(task_index, error, owner=holder)

    return Agent(
        engine=engine if engine is not None else LLMEngine(model, system=system or DURABLE_BLACKBOARD_GUIDANCE),
        tools=[
            *agents,
            Tool(set_plan),
            Tool(get_plan),
            Tool(add_tasks),
            Tool(cancel_task),
            Tool(set_task_schedule),
            Tool(claim_next),
            Tool(claim_task),
            Tool(mark_done),
            Tool(mark_failed),
        ],
        name=name,
        store=store,
        verbose=verbose,
    )
