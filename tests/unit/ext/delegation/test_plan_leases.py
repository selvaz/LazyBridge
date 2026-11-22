from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lazybridge import Envelope, Store
from lazybridge.ext.delegation import JobRegistry, make_plan_delegate
from lazybridge.ext.planners import durable_blackboard
from lazybridge.ext.planners.durable_blackboard import DurableBlackboard

_REQUEST = [{"task_index": 0, "expected_text": "draft", "objective": "draft"}]


class WaitingEngine:
    def __init__(self, result: str) -> None:
        self.result = result
        self.entered = asyncio.Event()
        self.finish = asyncio.Event()

    async def run(self, env, **kwargs):
        self.entered.set()
        await self.finish.wait()
        return Envelope(payload=self.result)


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(durable_blackboard, "time", SimpleNamespace(time=lambda: clock.now))
    return clock


def plan_worker(board, registry, tasks, factory, *, notify=None):
    return make_plan_delegate(
        engine_factory=factory,
        registry=registry,
        background_tasks=tasks,
        board=board,
        owner="shared",
        notify=notify,
        structured_outcomes=True,
    )


async def test_live_plan_worker_renews_over_multiple_intervals(tmp_path, monkeypatch, clock):
    path = str(tmp_path / "jobs.db")
    store = Store(db=path)
    board = DurableBlackboard(store, "plan", lease_seconds=0.15)
    other = DurableBlackboard(Store(db=path), "plan", lease_seconds=0.01)
    board.set_plan("work", ["draft"])
    registry = JobRegistry(store)
    engine = WaitingEngine("ready for review")
    tasks = set()
    worker = plan_worker(board, registry, tasks, lambda: engine)
    renewals = asyncio.Queue()
    refresh = board._refresh_delegation

    def record_renewal(*args, finished=False):
        ok = refresh(*args, finished=finished)
        if ok and not finished:
            renewals.put_nowait(board.snapshot().tasks[0]["claimed_at"])
        return ok

    monkeypatch.setattr(board, "_refresh_delegation", record_renewal)
    outcome = await worker.run(delegations=_REQUEST)
    job_id = outcome["items"][0]["job_id"]
    [running] = tasks
    try:
        await asyncio.wait_for(engine.entered.wait(), 2)
        original_deadline = other.snapshot().tasks[0]["delegation_lease_until"]

        async def wait_for_renewal():
            while await renewals.get() < clock.now:
                pass

        for _ in range(3):
            clock.now += 0.06
            # The contender has a different lease configuration and must
            # still honor the deadline in the shared task.
            assert other.claim_next("shared") is None
            assert isinstance(other.claim_task(0, "draft", owner="shared"), str)
            await asyncio.wait_for(wait_for_renewal(), 2)
        assert clock.now > original_deadline
        assert other.snapshot().tasks[0]["attempts"] == 1
        assert other.mark_done(0, "too early", owner="shared").startswith("REJECTED")
    finally:
        engine.finish.set()
        await asyncio.wait_for(running, 2)

    record = registry.find(job_id)
    assert record["status"] == "done" and record["delegation_id"] == job_id
    assert not other.mark_done(0, record["result"], owner="shared", delegation_id=job_id).startswith("REJECTED")


async def test_expired_worker_cannot_publish_after_same_owner_takeover(tmp_path, clock):
    path = str(tmp_path / "jobs.db")
    store = Store(db=path)
    other_store = Store(db=path)
    board = DurableBlackboard(store, "plan", lease_seconds=90)
    other = DurableBlackboard(other_store, "plan", lease_seconds=90)
    board.set_plan("work", ["draft"])
    registry = JobRegistry(store)
    first = WaitingEngine("stale result")
    second = WaitingEngine("current result")
    first_tasks, second_tasks, notices = set(), set(), []
    first_tool = plan_worker(board, registry, first_tasks, lambda: first, notify=notices.append)
    second_tool = plan_worker(other, JobRegistry(other_store), second_tasks, lambda: second)

    outcome = await first_tool.run(delegations=_REQUEST)
    first_id = outcome["items"][0]["job_id"]
    [first_task] = first_tasks
    await asyncio.wait_for(first.entered.wait(), 2)
    clock.now += 91
    outcome = await second_tool.run(delegations=_REQUEST)
    second_id = outcome["items"][0]["job_id"]
    [second_task] = second_tasks
    try:
        await asyncio.wait_for(second.entered.wait(), 2)
        first.finish.set()
        await asyncio.wait_for(first_task, 2)
        expired = registry.find(first_id)
        assert expired["status"] == "failed" and "lease expired" in expired["error"]
        assert "result" not in expired
        assert all("stale result" not in notice for notice in notices)
        assert board.mark_failed(0, "stale failure", owner="shared", delegation_id=first_id).startswith("REJECTED")
        assert board.snapshot().tasks[0]["delegation_id"] == second_id
    finally:
        first.finish.set()
        second.finish.set()
        await asyncio.wait_for(asyncio.gather(first_task, second_task), 2)

    assert registry.find(second_id)["result"] == "current result"
    assert board.mark_done(0, "stale review", owner="shared", delegation_id=first_id).startswith("REJECTED")
    assert not board.mark_done(0, "reviewed", owner="shared", delegation_id=second_id).startswith("REJECTED")


async def test_stale_default_setup_rollback_preserves_replacement(tmp_path, clock):
    path = str(tmp_path / "jobs.db")
    store = Store(db=path)
    board = DurableBlackboard(store, "plan", lease_seconds=90)
    other = DurableBlackboard(Store(db=path), "plan", lease_seconds=90)
    board.set_plan("work", ["draft"])
    registry = JobRegistry(store)
    tasks = set()

    def superseded_factory():
        clock.now += 91
        assert other.claim_task(0, "draft", owner="shared", delegation_id="replacement") == (0, "draft")
        raise RuntimeError("late setup failure")

    outcome = await plan_worker(board, registry, tasks, superseded_factory).run(delegations=_REQUEST)
    [task] = tasks
    await asyncio.wait_for(task, 2)
    job = registry.find(outcome["items"][0]["job_id"])
    assert job["status"] == "failed" and job["execution_started"] is False
    replacement = board.snapshot().tasks[0]
    assert replacement["status"] == "claimed" and replacement["delegation_id"] == "replacement"


async def test_lost_lease_cancels_a_waiting_worker(tmp_path, clock):
    store = Store(db=str(tmp_path / "jobs.db"))
    board = DurableBlackboard(store, "plan", lease_seconds=0.15)
    board.set_plan("work", ["draft"])
    registry = JobRegistry(store)
    engine = WaitingEngine("unreachable")
    tasks = set()

    outcome = await plan_worker(board, registry, tasks, lambda: engine).run(delegations=_REQUEST)
    [task] = tasks
    await asyncio.wait_for(engine.entered.wait(), 2)
    clock.now += 0.16
    await asyncio.wait_for(task, 2)

    expired = registry.find(outcome["items"][0]["job_id"])
    assert expired["status"] == "failed" and "lease expired" in expired["error"]
    assert "result" not in expired
    assert board.claim_next("replacement") == (0, "draft")
