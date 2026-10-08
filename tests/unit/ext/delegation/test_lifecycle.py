from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lazybridge import Store
from lazybridge.ext.delegation import JobContext, JobRegistry, JobRunner, make_background_delegate


class Reservation:
    allowed = True

    def __init__(self):
        self.released = 0
        self.refunded = 0

    def release(self):
        self.released += 1

    def refund(self):
        self.refunded += 1


@pytest.mark.parametrize("phase", ["prepare", "register", "execute", "finalize"])
async def test_each_phase_rolls_back_once_and_settles(tmp_path, phase):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write(
        "job", "work", tool_name="delegate", status="running", execution_started=False, extra={"future": "kept"}
    )
    grant = Reservation()
    context = JobContext("job", "work", "delegate", "worker", lambda: None, registry, admission=grant)
    rolled_back = []

    def fail(ctx):
        ctx.metadata["partial_setup"] = True
        raise RuntimeError("injected")

    runner = JobRunner(
        prepare=lambda ctx: None,
        execute=lambda ctx: None,
        finalize=lambda ctx: None,
        rollback=lambda ctx: rolled_back.append((ctx.phase, ctx.started)),
    )
    setattr(runner, phase, fail)
    with pytest.raises(RuntimeError, match="injected"):
        await runner(context)
    started = phase in ("execute", "finalize")
    assert rolled_back == [(phase, started)]
    assert (grant.released, grant.refunded) == ((1, 0) if started else (0, 1))
    assert registry.find("job")["status"] == "failed"
    assert registry.find("job")["future"] == "kept"


@pytest.mark.parametrize("winner", ["interrupted", "running"])
async def test_cas_loser_leaves_winning_record_and_refunds(tmp_path, winner):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write("job", "work", tool_name="delegate", status=winner, execution_started=True, extra={"winner": True})
    before = registry.find("job")
    grant = Reservation()
    rollback = []
    runner = JobRunner(
        prepare=lambda ctx: None,
        execute=lambda ctx: pytest.fail("must not execute"),
        rollback=lambda ctx: rollback.append(ctx.phase),
    )
    await runner(JobContext("job", "work", "delegate", "worker", lambda: None, registry, admission=grant))
    assert registry.find("job") == before
    assert rollback == ["register"]
    assert (grant.released, grant.refunded) == (0, 1)


async def test_default_phases_prepare_agent_before_cas_and_preserve_metadata(tmp_path, monkeypatch):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write(
        "job", "work", tool_name="delegate", status="running", execution_started=False, extra={"provenance": "caller"}
    )

    class Agent:
        def __init__(self, **kwargs):
            assert registry.find("job")["execution_started"] is False

        async def run(self, objective):
            assert registry.find("job")["execution_started"] is True
            return SimpleNamespace(ok=True, text=lambda: "done")

    monkeypatch.setattr("lazybridge.Agent", Agent)
    await JobRunner()(JobContext("job", "work", "delegate", "worker", lambda: object(), registry))
    record = registry.find("job")
    assert (record["status"], record["result"], record["provenance"]) == ("done", "done", "caller")


async def test_cancelled_execute_rolls_back_and_releases_even_if_rollback_raises(tmp_path):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write("job", "work", tool_name="delegate", status="running")
    grant = Reservation()
    entered = asyncio.Event()
    calls = []

    async def execute(ctx):
        entered.set()
        await asyncio.Event().wait()

    def rollback(ctx):
        calls.append(ctx.phase)
        raise ValueError("rollback failed")

    runner = JobRunner(prepare=lambda ctx: None, execute=execute, rollback=rollback)
    task = asyncio.create_task(
        runner(JobContext("job", "work", "delegate", "worker", lambda: None, registry, admission=grant))
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(ValueError, match="rollback failed"):
        await task
    assert calls == ["execute"]
    assert (grant.released, grant.refunded) == (1, 0)
    assert registry.find("job")["status"] == "failed"


async def test_background_accepts_runner(tmp_path):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    tasks = set()
    phases = []
    runner = JobRunner(
        prepare=lambda ctx: phases.append("prepare"),
        execute=lambda ctx: phases.append("execute"),
        finalize=lambda ctx: registry.update(ctx.job_id, {"status": "done"}),
    )
    tool = make_background_delegate(
        tool_name="work",
        label="worker",
        engine_factory=lambda: None,
        registry=registry,
        background_tasks=tasks,
        notify=None,
        doc="work",
        job_runner=runner,
    )
    await tool.run(objective="work")
    await asyncio.gather(*tasks)
    assert phases == ["prepare", "execute"]
