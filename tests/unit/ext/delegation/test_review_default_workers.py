from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lazybridge import Store
from lazybridge.ext.delegation import JobRegistry, background, make_parallel_delegate, make_plan_delegate
from lazybridge.ext.planners.durable_blackboard import DurableBlackboard
from tests.unit.ext.delegation.test_lifecycle import Reservation


async def drain(tasks):
    while tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)


def make_tool(kind, store, grant, tasks, factory):
    registry = JobRegistry(store)
    board = DurableBlackboard(store, "plan")
    board.set_plan("work", ["task"])

    async def admission_gate():
        return grant

    kwargs = dict(engine_factory=factory, registry=registry, background_tasks=tasks, admission_gate=admission_gate)
    if kind == "parallel":
        tool = make_parallel_delegate(**kwargs, doc="parallel")
        arguments = dict(objectives=["work"])
    else:
        tool = make_plan_delegate(**kwargs, board=board, owner="caller")
        arguments = dict(delegations=[dict(task_index=0, expected_text="task", objective="work")])
    return tool, registry, board, arguments


@pytest.mark.parametrize("kind", ["parallel", "plan"])
@pytest.mark.parametrize("outcome", ["engine", "agent", "execution_error", "success", "execution_cancel"])
async def test_default_admission_settles_by_execution_boundary(tmp_path, monkeypatch, kind, outcome):
    grant = Reservation()
    tasks = set()
    entered = asyncio.Event()

    class Agent:
        def __init__(self, **kwargs):
            if outcome == "agent":
                raise RuntimeError("Agent setup failed")

        async def run(self, objective):
            entered.set()
            if outcome == "execution_error":
                raise RuntimeError("execution failed")
            if outcome == "execution_cancel":
                await asyncio.Event().wait()
            return SimpleNamespace(ok=True, text=lambda: "done")

    def factory():
        if outcome == "engine":
            raise RuntimeError("engine setup failed")
        return object()

    monkeypatch.setattr("lazybridge.Agent", Agent)
    store = Store(db=str(tmp_path / "jobs.db"))
    tool, registry, _, arguments = make_tool(kind, store, grant, tasks, factory)
    await tool.run(**arguments)
    if outcome == "execution_cancel":
        await entered.wait()
        [task] = tasks
        task.cancel()
    await drain(tasks)
    started = outcome not in ("engine", "agent")
    assert (grant.released, grant.refunded) == ((1, 0) if started else (0, 1))
    [record] = [value for _, value in store.items(prefix=registry._prefix)]
    assert record["execution_started"] is started
    assert record["status"] == ("done" if outcome == "success" else "failed")


@pytest.mark.parametrize("kind", ["parallel", "plan"])
@pytest.mark.parametrize("with_admission", [False, True])
async def test_default_pre_entry_cancel_closes_job_refunds_and_finishes_record(
    tmp_path, monkeypatch, recwarn, kind, with_admission
):
    grant = Reservation() if with_admission else None
    tasks = set()
    inner = []
    original = background._run_delegate_job

    def capture(*args, **kwargs):
        coroutine = original(*args, **kwargs)
        inner.append(coroutine)
        return coroutine

    monkeypatch.setattr(background, "_run_delegate_job", capture)
    store = Store(db=str(tmp_path / "jobs.db"))
    tool, registry, board, arguments = make_tool(
        kind, store, grant, tasks, lambda: pytest.fail("cancelled job must never construct an engine")
    )
    await tool.run(**arguments)
    [task] = tasks
    task.cancel()  # deliberately no intervening await or sleep
    await drain(tasks)
    assert inner[0].cr_frame is None
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    if grant:
        assert (grant.released, grant.refunded) == (0, 1)
    [record] = [value for _, value in store.items(prefix=registry._prefix)]
    assert record["status"] == "failed" and record["execution_started"] is False
    if kind == "plan":
        assert board.snapshot().tasks[0]["status"] == "todo"
        assert board.snapshot().tasks[0]["owner"] is None
