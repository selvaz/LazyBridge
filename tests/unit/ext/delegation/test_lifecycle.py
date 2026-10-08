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


async def test_register_failure_after_cas_refunds_and_rolls_back(tmp_path):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write("job", "work", tool_name="delegate", status="running", execution_started=False)
    grant = Reservation()
    calls = []

    def register(context):
        assert context.begin_execution()
        raise RuntimeError("after CAS")

    runner = JobRunner(
        prepare=lambda ctx: None,
        register=register,
        rollback=lambda ctx: calls.append((ctx.phase, ctx.registered, ctx.started)),
    )
    with pytest.raises(RuntimeError, match="after CAS"):
        await runner(JobContext("job", "work", "delegate", "worker", object, registry, admission=grant))
    assert calls == [("register", True, False)]
    assert (grant.released, grant.refunded) == (0, 1)
    assert registry.find("job")["status"] == "failed"
    assert registry.find("job")["execution_started"] is False


async def test_setup_failure_cannot_overwrite_another_started_worker(tmp_path):
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    registry.write("job", "work", tool_name="delegate", status="running", execution_started=False)

    def prepare(context):
        assert registry.begin_execution("job")  # a competing worker wins during preparation
        raise RuntimeError("local setup failed")

    with pytest.raises(RuntimeError, match="local setup failed"):
        await JobRunner(prepare=prepare)(JobContext("job", "work", "delegate", "worker", object, registry))
    assert registry.find("job")["status"] == "running"
    assert registry.find("job")["execution_started"] is True


async def test_background_failure_is_recorded_and_the_task_ends_quietly(tmp_path):
    """JobRunner re-raises for a caller awaiting it; a background delegate's own
    retained task must not, or every failed job resurfaces later as asyncio's
    'Task exception was never retrieved'. The failure stays in the record."""
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))
    tasks = set()

    def fail(ctx):
        raise RuntimeError("no worker was built")

    tool = make_background_delegate(
        tool_name="work",
        label="worker",
        engine_factory=lambda: None,
        registry=registry,
        background_tasks=tasks,
        notify=None,
        doc="work",
        job_runner=JobRunner(prepare=fail),
    )
    await tool.run(objective="work")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert results == [None]
    [record] = [v for _, v in registry._store.items(prefix=registry._prefix)]
    assert record["status"] == "failed" and "no worker was built" in record["error"]
