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

import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from lazybridge import Agent, LLMEngine, Store, Tool

TaskStatus = Literal["todo", "claimed", "done", "failed"]

#: Bumped when the persisted document shape changes incompatibly.
BLACKBOARD_VERSION = 1

DURABLE_BLACKBOARD_GUIDANCE = """\
# How to work

You keep a durable plan that survives restarts. It is the only place your
progress is recorded — your own memory of this conversation is not.

Tools:

- ``get_plan()``                       — read the plan and see what is left.
- ``set_plan(reasoning, tasks)``       — create the plan (only when there is none).
- ``claim_next()``                     — take the next task; returns its index and text.
- ``mark_done(task_index, summary)``   — close a task with a 1-3 sentence result.
- ``mark_failed(task_index, error)``   — give a task back after a real failure.

## Workflow

1. Always start with ``get_plan()``.
2. If there is no plan, call ``set_plan`` with 3-6 coarse, self-contained
   tasks in execution order, then stop and report the plan.
3. If there is a plan, call ``claim_next()``. If it returns a task: do **that
   one task only**, then ``mark_done`` it and report what you did. Do not
   claim a second task in the same run.
4. If ``claim_next()`` says the plan is complete, report the final result.
5. If the work genuinely fails, call ``mark_failed`` with the reason —
   the task returns to the queue and is retried on a later run.

Never invent progress: a task counts as done only once ``mark_done`` returns.
"""


@dataclass(frozen=True)
class BlackboardSnapshot:
    """Read-only view of a plan, for callers that want data instead of text."""

    plan_id: str
    reasoning: str
    tasks: list[dict[str, Any]]

    @property
    def complete(self) -> bool:
        return bool(self.tasks) and all(t["status"] in ("done", "failed") for t in self.tasks)

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
                "tasks": [
                    {
                        "text": str(t),
                        "status": "todo",
                        "result": "",
                        "error": "",
                        "attempts": 0,
                        "owner": None,
                        "claimed_at": None,
                    }
                    for t in tasks
                ],
            }
            return fresh, self._render(fresh)

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
        )

    def render(self) -> str:
        """Human/LLM-readable state, including which task is next."""
        return self._render(self._read())

    def _render(self, doc: dict[str, Any] | None) -> str:
        if doc is None or not doc.get("tasks"):
            return "(no plan yet; call set_plan)"
        marks = {"todo": "[ ]", "claimed": "[~]", "done": "[x]", "failed": "[!]"}
        lines = [f"plan: {doc.get('plan_id')}", f"reasoning: {doc.get('reasoning', '')}"]
        for i, task in enumerate(doc["tasks"]):
            row = f"  {i}. {marks.get(task['status'], '[?]')} {task['text']}"
            if task.get("result"):
                row += f"\n       → {task['result']}"
            if task.get("error"):
                row += f"\n       ! {task['error']} (attempts: {task.get('attempts', 0)})"
            lines.append(row)
        snapshot = self._snapshot_of(doc)
        if snapshot.complete:
            done = sum(1 for t in snapshot.tasks if t["status"] == "done")
            failed = len(snapshot.tasks) - done
            lines.append(
                f"plan complete — {done} done, {failed} failed" if failed else "plan complete — all tasks done"
            )
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
            index = next((i for i, t in enumerate(tasks) if t["status"] == "todo"), None)
            if index is None:
                index = next(
                    (
                        i
                        for i, t in enumerate(tasks)
                        if t["status"] == "claimed"
                        and t.get("claimed_at") is not None
                        and now - float(t["claimed_at"]) > self.lease_seconds
                    ),
                    None,
                )
            if index is None:
                return None, None
            task = tasks[index]
            if task.get("attempts", 0) >= self.max_attempts:
                # Out of retries: park it so the plan can finish instead of
                # handing the same poison task out forever.
                task.update(status="failed", owner=None, claimed_at=None)
                task["error"] = task.get("error") or f"exhausted {self.max_attempts} attempts"
                new_doc = {**doc, "tasks": tasks}
                return new_doc, None
            task.update(status="claimed", owner=holder, claimed_at=now, attempts=task.get("attempts", 0) + 1)
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
                task.update(status="done", result=text, error="", owner=None, claimed_at=None)
            else:
                exhausted = task.get("attempts", 0) >= self.max_attempts
                task.update(status="failed" if exhausted else "todo", error=text, owner=None, claimed_at=None)
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

    def mark_done(task_index: int, result_summary: str) -> str:
        """Close a task with a 1-3 sentence summary of what was produced."""
        return board.mark_done(task_index, result_summary, owner=holder)

    def mark_failed(task_index: int, error: str) -> str:
        """Give a task back after a real failure; it is retried on a later run."""
        return board.mark_failed(task_index, error, owner=holder)

    return Agent(
        engine=engine if engine is not None else LLMEngine(model, system=system or DURABLE_BLACKBOARD_GUIDANCE),
        tools=[*agents, Tool(set_plan), Tool(get_plan), Tool(claim_next), Tool(mark_done), Tool(mark_failed)],
        name=name,
        store=store,
        verbose=verbose,
    )
