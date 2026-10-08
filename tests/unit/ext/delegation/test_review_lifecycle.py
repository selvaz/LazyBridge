from __future__ import annotations

import asyncio
import inspect

import pytest

from lazybridge import Store
from lazybridge.ext.delegation import JobContext, JobRegistry, JobRunner, make_background_delegate, make_plan_delegate
from tests.unit.ext.delegation.test_lifecycle import Reservation


@pytest.mark.parametrize("failure", ["engine", "agent", "execute"])
@pytest.mark.parametrize("notify_raises", [False, True])
async def test_runner_exception_notifies_once_and_does_not_break_cleanup(tmp_path, monkeypatch, failure, notify_raises):
    store = Store(db=str(tmp_path / "jobs.db"))
    registry = JobRegistry(store)
    grant = Reservation()
    notices = []

    def notify(message):
        notices.append(message)
        if notify_raises:
            raise RuntimeError("notification channel down")

    class Agent:
        def __init__(self, **kwargs):
            if failure == "agent":
                raise ValueError("agent setup failed")

        async def run(self, objective):
            raise ValueError("execute failed")

    def factory():
        if failure == "engine":
            raise ValueError("engine setup failed")
        return object()

    monkeypatch.setattr("lazybridge.Agent", Agent)

    async def gate():
        return grant

    tasks = set()
    tool = make_background_delegate(
        tool_name="work",
        label="worker",
        engine_factory=factory,
        registry=registry,
        background_tasks=tasks,
        notify=notify,
        doc="work",
        admission_gate=gate,
    )
    await tool.run(objective="objective")
    await asyncio.gather(*tasks)  # the recorded failure ends the background task normally
    [job] = [value for _, value in store.items(prefix=registry._prefix)]
    before = " before it started" if failure != "execute" else ""
    error = f"{failure} setup failed" if failure != "execute" else "execute failed"
    assert [message for message in notices if "FAILED" in message] == [
        f"worker job {job['job_id'][:8]} FAILED{before}: objective\n\n{error}"
    ]
    assert job["status"] == "failed"
    assert (grant.released, grant.refunded) == ((1, 0) if failure == "execute" else (0, 1))


async def test_execution_cas_records_identity_and_unknown_cost_atomically(tmp_path, monkeypatch):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write(
        "job",
        "work",
        tool_name="delegate",
        status="running",
        execution_started=False,
        extra={"caller_provenance": "kept"},
    )
    cas_values = []
    original = registry._store.compare_and_swap

    def cas(key, expected, replacement):
        if replacement.get("execution_started") is True:
            cas_values.append(replacement)
        return original(key, expected, replacement)

    monkeypatch.setattr(registry._store, "compare_and_swap", cas)

    class Engine:
        model = "chosen-model"
        reasoning_effort = "high"

    class Agent:
        def __init__(self, *, engine, name):
            self.engine = engine
            assert registry.find("job")["execution_started"] is False

        async def run(self, objective):
            current = registry.find("job")
            assert current["cost_unknown"] is True
            assert (current["engine"], current["model"], current["effort"]) == ("Engine", "chosen-model", "high")
            raise RuntimeError("provider failed after spending")

    monkeypatch.setattr("lazybridge.Agent", Agent)
    with pytest.raises(RuntimeError, match="provider failed"):
        await JobRunner()(JobContext("job", "work", "delegate", "worker", Engine, registry))
    boundary = cas_values[0]
    assert boundary["execution_started"] is True and boundary["cost_unknown"] is True
    assert (boundary["engine"], boundary["model"], boundary["effort"]) == ("Engine", "chosen-model", "high")
    failed = registry.find("job")
    assert failed["status"] == "failed" and failed["cost_unknown"] is True
    assert failed["caller_provenance"] == "kept"
    assert (failed["engine"], failed["model"], failed["effort"]) == ("Engine", "chosen-model", "high")


def test_default_phase_callbacks_are_unbound_instance_fields():
    for name in ("prepare", "register", "execute", "finalize"):
        runner = JobRunner()
        assert name not in JobRunner.__dict__
        callback = runner.__dict__[name]
        assert list(inspect.signature(callback).parameters) == ["context"]


async def test_plan_scheduling_cancellation_is_cleaned_up_and_re_raised(tmp_path, monkeypatch):
    from lazybridge.ext.planners.durable_blackboard import DurableBlackboard

    store = Store(db=str(tmp_path / "jobs.db"))
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan")
    board.set_plan("work", ["task"])
    grant = Reservation()

    async def gate():
        return grant

    def cancelled_scheduler(tasks, coroutine):
        raise asyncio.CancelledError

    monkeypatch.setattr("lazybridge.ext.delegation.background._track", cancelled_scheduler)
    tool = make_plan_delegate(
        engine_factory=object,
        registry=registry,
        background_tasks=set(),
        board=board,
        owner="caller",
        admission_gate=gate,
        structured_outcomes=True,
    )
    with pytest.raises(asyncio.CancelledError):
        assert await tool.run(delegations=[dict(task_index=0, expected_text="task", objective="work")]) is None
    assert (grant.released, grant.refunded) == (0, 1)
    [job] = [value for _, value in store.items(prefix=registry._prefix)]
    assert job["status"] == "failed" and job["execution_started"] is False
    assert board.snapshot().tasks[0]["status"] == "todo"
