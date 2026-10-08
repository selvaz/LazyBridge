from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lazybridge import Store
from lazybridge.ext.delegation import JobRegistry, JobRunner, make_parallel_delegate, make_plan_delegate
from lazybridge.ext.planners.durable_blackboard import DurableBlackboard
from tests.unit.ext.delegation.test_lifecycle import Reservation


def restore_claim(board, context):
    """Test caller's owner/attempt-protected CAS rollback; no CEO policy."""
    if not context.claimed:
        return
    current = board.store.read(board.key)
    tasks = [dict(task) for task in current["tasks"]]
    task = tasks[context.task_index]
    if task["owner"] != context.claim_owner or task["attempts"] != context.attempts_before + 1:
        return
    task.update(status="todo", owner=None, claimed_at=None, attempts=context.attempts_before)
    assert board.store.compare_and_swap(board.key, current, {**current, "tasks": tasks})


async def test_five_items_third_refused_exact_outcome_and_attempts(tmp_path, monkeypatch):
    store = Store(db=str(tmp_path / "jobs.db"))
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan")
    names = [f"task {n}" for n in range(1, 6)]
    board.set_plan("work", names)
    # Previous real attempts must survive a batch refusal exactly.
    current = store.read(board.key)
    store.write(board.key, {**current, "tasks": [{**t, "attempts": 1} for t in current["tasks"]]})
    grants = [Reservation(), Reservation()]
    refusal = SimpleNamespace(allowed=False, rejection_text=lambda: "REFUSED: quota")
    decisions = iter([*grants, refusal])
    admission_calls = []
    rolled_back = []
    running = []
    finish = asyncio.Event()

    async def gate():
        admission_calls.append(1)
        return next(decisions)

    class Agent:
        def __init__(self, **kwargs):
            pass

        async def run(self, objective):
            running.append(objective)
            await finish.wait()
            return SimpleNamespace(ok=True, text=lambda: "done")

    monkeypatch.setattr("lazybridge.Agent", Agent)

    async def rollback(context):
        rolled_back.append(
            (context.task_index, context.attempts_before, context.claim_owner, context.claimed, context.started)
        )
        restore_claim(board, context)

    tasks = set()
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=tasks,
        board=board,
        owner="caller",
        admission_gate=gate,
        rollback_claim=rollback,
        structured_outcomes=True,
        stop_on_refusal=True,
    )
    result = await tool.func([dict(task_index=i, expected_text=name, objective=name) for i, name in enumerate(names)])
    jobs = {v["task_index"]: v for _, v in store.items(prefix=registry._prefix)}
    assert result == {
        "items": [
            dict(item_number=1, task_index=0, status="started", job_id=jobs[0]["job_id"], reason=None),
            dict(item_number=2, task_index=1, status="started", job_id=jobs[1]["job_id"], reason=None),
            dict(item_number=3, task_index=2, status="refused", job_id=None, reason="REFUSED: quota"),
            dict(
                item_number=4, task_index=3, status="refused", job_id=None, reason="not started after refusal at item 3"
            ),
            dict(
                item_number=5, task_index=4, status="refused", job_id=None, reason="not started after refusal at item 3"
            ),
        ]
    }
    assert len(admission_calls) == 3
    assert rolled_back == [(i, 1, "caller", False, False) for i in (2, 3, 4)]
    assert [t["attempts"] for t in board.snapshot().tasks] == [2, 2, 1, 1, 1]
    await asyncio.sleep(0)
    assert running == names[:2]
    assert len(tasks) == 2 and all(not task.done() for task in tasks)
    finish.set()
    await asyncio.gather(*tasks)
    assert all((g.released, g.refunded) == (1, 0) for g in grants)


@pytest.mark.parametrize("failure", ["write", "schedule", "engine", "agent", "cas", "immediate_cancel", "execute"])
async def test_claim_attempt_restored_for_every_unstarted_path(tmp_path, monkeypatch, failure):
    store = Store(db=str(tmp_path / "jobs.db"))
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan")
    board.set_plan("work", ["task"])
    grant = Reservation()
    callbacks = []
    runner_rollbacks = []

    async def gate():
        return grant

    def rollback(context):
        callbacks.append((context.phase, context.attempts_before, context.claim_owner, context.job_id))
        restore_claim(board, context)

    class Agent:
        def __init__(self, **kwargs):
            if failure == "agent":
                raise RuntimeError("agent setup")

        async def run(self, objective):
            if failure == "execute":
                raise RuntimeError("execution failed")
            return SimpleNamespace(ok=True, text=lambda: "done")

    def factory():
        if failure == "engine":
            raise RuntimeError("engine setup")
        return object()

    monkeypatch.setattr("lazybridge.Agent", Agent)
    if failure == "write":
        original = registry.write

        def write(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("write failed")

        monkeypatch.setattr(registry, "write", write)
    elif failure == "schedule":

        def track(*args):
            raise RuntimeError("schedule failed")

        monkeypatch.setattr("lazybridge.ext.delegation.background._track", track)
    tasks = set()
    runner = JobRunner(rollback=lambda ctx: runner_rollbacks.append(ctx.started))
    tool = make_plan_delegate(
        engine_factory=factory,
        registry=registry,
        background_tasks=tasks,
        board=board,
        owner="caller",
        admission_gate=gate,
        rollback_claim=rollback,
        job_runner=runner,
        structured_outcomes=True,
    )
    result = await tool.func([dict(task_index=0, expected_text="task", objective="work")])
    [job] = [v for _, v in store.items(prefix=registry._prefix)]
    if failure == "cas":
        registry.update(job["job_id"], {"status": "interrupted"})
    elif failure == "immediate_cancel":
        [task] = tasks
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)
    await asyncio.gather(*tasks, return_exceptions=True)
    started = failure == "execute"
    assert runner_rollbacks == [started]
    assert len(callbacks) == (0 if started else 1)
    if callbacks:
        assert callbacks[0][1:] == (0, "caller", job["job_id"])
    assert board.snapshot().tasks[0]["attempts"] == (1 if started else 0)
    assert board.snapshot().tasks[0]["status"] == ("claimed" if started else "todo")
    assert (grant.released, grant.refunded) == ((1, 0) if started else (0, 1))
    assert registry.find(job["job_id"])["status"] in ("failed", "interrupted")
    assert result["items"][0]["status"] == ("failed" if failure in ("write", "schedule") else "started")


async def test_rollback_uses_captured_owner_after_takeover(tmp_path, monkeypatch):
    store = Store(db=str(tmp_path / "jobs.db"))
    board = DurableBlackboard(store, "plan")
    board.set_plan("work", ["task"])
    seen = []

    def prepare(context):
        current = store.read(board.key)
        store.write(board.key, {**current, "tasks": [{**current["tasks"][0], "owner": "new-owner"}]})
        raise RuntimeError("setup failed")

    def rollback(context):
        seen.append(context.claim_owner)
        restore_claim(board, context)

    tasks = set()
    tool = make_plan_delegate(
        engine_factory=object,
        registry=JobRegistry(store),
        background_tasks=tasks,
        board=board,
        owner="original",
        rollback_claim=rollback,
        job_runner=JobRunner(prepare=prepare),
    )
    await tool.func([dict(task_index=0, expected_text="task", objective="work")])
    await asyncio.gather(*tasks, return_exceptions=True)
    assert seen == ["original"]
    task = board.snapshot().tasks[0]
    assert task["owner"] == "new-owner" and task["attempts"] == 1


async def test_parallel_structured_partial_scheduling_failure(tmp_path, monkeypatch):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    tasks = set()
    original = registry.write

    def write(job_id, objective, **kwargs):
        if objective == "bad" and kwargs["status"] == "running":
            raise RuntimeError("write unavailable")
        original(job_id, objective, **kwargs)

    monkeypatch.setattr(registry, "write", write)
    runner = JobRunner(
        prepare=lambda ctx: None,
        execute=lambda ctx: None,
        finalize=lambda ctx: registry.update(ctx.job_id, {"status": "done"}),
    )
    tool = make_parallel_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=tasks,
        doc="parallel",
        job_runner=runner,
        structured_outcomes=True,
    )
    result = await tool.func(["one", "bad", "three"])
    assert [r["status"] for r in result["items"]] == ["started", "failed", "started"]
    assert result["items"][1]["reason"] == "run_parallel scheduling failed: write unavailable"
    await asyncio.gather(*tasks)


@pytest.mark.parametrize(
    "count,in_flight,code",
    [(0, 0, "empty"), (3, 0, "batch_cap"), (1, 2, "in_flight_cap")],
)
async def test_whole_batch_refusal_carries_a_stable_code(tmp_path, count, in_flight, code):
    """Callers tell a whole-batch refusal from per-item ones by its code, not by
    matching the wording -- which is free to change."""
    store = Store(db=str(tmp_path / "jobs.db"))
    board = DurableBlackboard(store, "plan")
    board.set_plan("work", ["a", "b", "c"])
    tasks: set[asyncio.Task] = {asyncio.ensure_future(asyncio.sleep(10)) for _ in range(in_flight)}
    tool = make_plan_delegate(
        engine_factory=object,
        registry=JobRegistry(store),
        background_tasks=tasks,
        board=board,
        owner="caller",
        structured_outcomes=True,
        max_parallel_objectives=2,
        max_in_flight_delegate_tasks=2,
    )
    try:
        delegations = [dict(task_index=i, expected_text=t, objective=t) for i, t in enumerate(["a", "b", "c"][:count])]
        result = await tool.func(delegations)
    finally:
        for task in tasks:
            task.cancel()
    assert result["batch_refused"]["code"] == code
    assert result["batch_refused"]["reason"].startswith("REJECTED:")
    assert all(item["status"] == "refused" for item in result["items"])
