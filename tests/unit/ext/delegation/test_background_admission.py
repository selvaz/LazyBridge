from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lazybridge import Store
from lazybridge.ext.delegation import JobRegistry, JobRunner, make_background_delegate, make_claude_writer
from tests.unit.ext.delegation.test_lifecycle import Reservation


def build(tmp_path, gate, *, runner=None, factory=lambda: None):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    tasks = set()
    tool = make_background_delegate(
        tool_name="work",
        label="worker",
        engine_factory=factory,
        registry=registry,
        background_tasks=tasks,
        notify=None,
        doc="work",
        admission_gate=gate,
        job_runner=runner,
    )
    return tool, registry, tasks


@pytest.mark.parametrize("path", ["success", "engine", "agent", "execute", "cas", "cancel"])
async def test_no_confirm_reservation_parity(tmp_path, monkeypatch, path):
    grant = Reservation()
    calls = []
    entered = asyncio.Event()

    async def gate():
        calls.append("admission")
        return grant

    class Agent:
        def __init__(self, **kwargs):
            calls.append("agent")
            if path == "agent":
                raise RuntimeError("agent setup")

        async def run(self, objective):
            calls.append("execute")
            entered.set()
            if path == "execute":
                raise RuntimeError("execution")
            if path == "cancel":
                await asyncio.Event().wait()
            return SimpleNamespace(ok=True, text=lambda: "done")

    def factory():
        calls.append("engine")
        if path == "engine":
            raise RuntimeError("engine setup")
        return None

    monkeypatch.setattr("lazybridge.Agent", Agent)
    tool, registry, tasks = build(tmp_path, gate, factory=factory)
    await tool.func("objective")
    assert calls == ["admission"]  # scheduling reserves, execution is still deferred
    [record] = [v for _, v in registry._store.items(prefix=registry._prefix)]
    if path == "cas":
        registry.update(record["job_id"], {"status": "interrupted"})
    scheduled = list(tasks)
    if path == "cancel":
        await entered.wait()
        scheduled[0].cancel()
    await asyncio.gather(*scheduled, return_exceptions=True)
    started = path in ("success", "execute", "cancel")
    assert (grant.released, grant.refunded) == ((1, 0) if started else (0, 1))
    assert registry.find(record["job_id"])["status"] in ("done", "failed", "interrupted")


@pytest.mark.parametrize("failure", ["write", "schedule", "immediate_cancel"])
async def test_not_scheduled_or_never_entered_refunds_once(tmp_path, monkeypatch, failure):
    grant = Reservation()
    rollbacks = []

    async def gate():
        return grant

    runner = JobRunner(rollback=lambda ctx: rollbacks.append(ctx.phase))
    tool, registry, tasks = build(tmp_path, gate, runner=runner)
    if failure == "write":
        original = registry.write

        def write_then_raise(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("write")

        monkeypatch.setattr(registry, "write", write_then_raise)
    elif failure == "schedule":

        def fail(*args):
            raise RuntimeError("schedule")

        monkeypatch.setattr("lazybridge.ext.delegation.background._track", fail)
    if failure == "immediate_cancel":
        await tool.func("work")
        [task] = tasks
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        await asyncio.gather(*tasks)
    else:
        with pytest.raises(RuntimeError, match=failure):
            await tool.func("work")
    assert (grant.released, grant.refunded) == (0, 1)
    assert rollbacks == ["schedule"]
    [record] = [v for _, v in registry._store.items(prefix=registry._prefix)]
    assert record["status"] == "failed"
    assert record["execution_started"] is False


async def test_claude_writer_admission_refuses_before_registration(tmp_path):
    refused = Reservation()
    refused.allowed = False
    refused.rejection_text = lambda: "REFUSED: caller quota"

    async def gate():
        return refused

    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    tasks = set()
    tool = make_claude_writer(
        workspace_root=tmp_path,
        gate=object(),
        registry=registry,
        background_tasks=tasks,
        doc="write",
        admission_gate=gate,
    )
    assert await tool.func("work") == "REFUSED: caller quota"
    assert tasks == set()
    assert list(registry._store.items(prefix=registry._prefix)) == []
    assert (refused.released, refused.refunded) == (0, 0)
