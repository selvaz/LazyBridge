"""Regression coverage for the round-two durable compatibility fixes."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from types import SimpleNamespace

import pytest

from lazybridge import Agent, Envelope, Store
from lazybridge import _process_owner as process_owner
from lazybridge._verify import verify_with_retry
from lazybridge.engines.plan import ConcurrentPlanRunError, Plan, Step
from lazybridge.envelope import ErrorInfo
from lazybridge.ext.delegation.admission import _AdmissionLease
from lazybridge.ext.delegation.jobs import JobRegistry
from lazybridge.ext.planners import durable_blackboard
from lazybridge.ext.planners.durable_blackboard import DurableBlackboard
from lazybridge.session import EventType, Session, _guarded_trace
from lazybridge.testing import MockAgent


def _plan(store: Store, **kwargs: object) -> Agent:
    engine = Plan(
        Step(MockAgent("finished", name="only"), name="only"),
        store=store,
        checkpoint_key="checkpoint",
        resume=True,
        **kwargs,
    )
    return Agent(engine=engine, name="checkpoint-plan")


@pytest.mark.asyncio
async def test_legacy_checkpoint_staleness_override_and_reconciliation(monkeypatch, caplog) -> None:
    now = 10_000.0
    monkeypatch.setattr("lazybridge.engines.plan._checkpoint.time", SimpleNamespace(time=lambda: now))
    monkeypatch.setattr(process_owner, "time", SimpleNamespace(time=lambda: now))
    recent = {"status": "running", "next_step": "only", "run_uid": "legacy", "updated_at": now - 899}
    store = Store()
    store.write("checkpoint", recent)
    with pytest.raises(ConcurrentPlanRunError, match="unverified active owner"):
        await _plan(store).run("work")

    store.write("checkpoint", {**recent, "updated_at": now - 901})
    with caplog.at_level(logging.WARNING):
        assert (await _plan(store).run("work")).ok
    assert "Adopting unowned active checkpoint" in caplog.text

    store.write("checkpoint", recent)
    assert (await _plan(store, adopt_unowned=True).run("work")).ok

    store.write("checkpoint", {**recent, "status": "uncertain"})
    plan = _plan(store).engine
    with pytest.raises(ConcurrentPlanRunError, match="requires reconciliation"):
        await Agent(engine=plan, name="checkpoint-plan").run("work")
    assert plan.clear_checkpoint() is True
    assert (await Agent(engine=plan, name="checkpoint-plan").run("work")).ok


def test_process_owner_pid_reuse_dead_pid_and_current_process(monkeypatch) -> None:
    foreign = {"owner_pid": os.getpid() + 10_000, "owner_boot_id": "boot", "owner_start_time": 10.0}
    monkeypatch.setattr(process_owner, "_pid_is_running", lambda _pid: True)
    monkeypatch.setattr(process_owner, "_process_start_time", lambda _pid: 11.0)
    assert process_owner.owner_is_alive(foreign) is False
    monkeypatch.setattr(process_owner, "_pid_is_running", lambda _pid: False)
    assert process_owner.owner_is_alive(foreign) is False
    assert process_owner.owner_is_alive(process_owner.current_owner_fields()) is True


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_inconclusive_platform_probe_allows_stale_reclaim(monkeypatch, caplog, platform) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(process_owner, "owner_is_alive", lambda _record: None)
    monkeypatch.setattr("lazybridge.ext.delegation.jobs.owner_is_alive", lambda _record: None)
    monkeypatch.setattr("lazybridge.ext.delegation.jobs.time", SimpleNamespace(time=lambda: 2_000.0))
    registry = JobRegistry(Store())
    registry.write("stale", "work", tool_name="delegate", status="running", created_at=1_000.0)
    with caplog.at_level(logging.WARNING):
        assert registry.reclaim_interrupted(stale_after=900) == ["stale"]
    assert "Reclaiming stale unowned delegation job" in caplog.text


def test_reclaim_unowned_uses_age_by_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("lazybridge.ext.delegation.jobs.time", SimpleNamespace(time=lambda: 2_000.0))
    monkeypatch.setattr(process_owner, "time", SimpleNamespace(time=lambda: 2_000.0))
    registry = JobRegistry(Store())
    registry.write("stale", "x", tool_name="delegate", status="running", created_at=1_000.0)
    registry.write("recent", "x", tool_name="delegate", status="running", created_at=1_500.0)
    assert registry.reclaim_interrupted(stale_after=900) == ["stale"]
    assert registry.find("recent")["status"] == "running"
    assert registry.reclaim_interrupted(reclaim_unowned=True) == ["recent"]


@pytest.mark.asyncio
async def test_verify_retries_ordinary_failure_but_not_uncertain() -> None:
    calls: list[Envelope] = []
    verdict_inputs: list[str] = []

    async def failed(env: Envelope) -> Envelope:
        calls.append(env)
        return Envelope(task=env.task, error=ErrorInfo(type="Failed", message="ordinary"))

    def judge(text: str) -> str:
        verdict_inputs.append(text)
        return "rejected: retry"

    result = await verify_with_retry(SimpleNamespace(), Envelope.from_task("work"), judge, max_verify=3, run=failed)
    assert result.error and result.error.type == "Failed"
    assert len(calls) == len(verdict_inputs) == 3
    assert "rejected: retry" in (calls[1].context or "")

    calls.clear()
    verdict_inputs.clear()

    async def uncertain(env: Envelope) -> Envelope:
        calls.append(env)
        return Envelope(error=ErrorInfo(type="Unknown", message="maybe ran", outcome_uncertain=True))

    result = await verify_with_retry(SimpleNamespace(), Envelope.from_task("work"), judge, max_verify=3, run=uncertain)
    assert result.error and result.error.outcome_uncertain
    assert len(calls) == 1 and verdict_inputs == []


def test_blackboard_missing_delegation_id_only_after_release_or_expiry(monkeypatch) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(durable_blackboard, "time", SimpleNamespace(time=lambda: clock.now))
    board = DurableBlackboard(Store(), "plan", lease_seconds=10)
    board.set_plan("reason", ["task"])
    assert board.claim_task(0, "task", owner="worker", delegation_id="job") == (0, "task")
    assert board.mark_done(0, "result", owner="worker").startswith("REJECTED")

    document = board.store.read(board.key)
    document["tasks"][0]["delegation_active"] = False
    board.store.write(board.key, document)
    assert not board.mark_done(0, "result", owner="worker").startswith("REJECTED")
    assert "delegation_id" not in board.snapshot().tasks[0]

    expired = DurableBlackboard(Store(), "expired", lease_seconds=10)
    expired.set_plan("reason", ["task"])
    expired.claim_task(0, "task", owner="worker", delegation_id="job")
    clock.now += 11
    assert not expired.mark_done(0, "result", owner="worker").startswith("REJECTED")


@pytest.mark.asyncio
async def test_admission_settlement_timeout_logs_and_propagates_cancellation(caplog) -> None:
    blocker = asyncio.Event()

    class Decision:
        allowed = True

        async def release(self) -> None:
            await blocker.wait()

    lease = _AdmissionLease(Decision(), settlement_timeout=0.02)
    task = asyncio.create_task(lease.release())
    await asyncio.sleep(0)
    task.cancel()
    with caplog.at_level(logging.ERROR), pytest.raises(asyncio.CancelledError):
        await task
    assert "admission settlement exceeded 0.0 seconds" in caplog.text
    assert lease._settlement is not None and not lease._settlement.done()
    blocker.set()
    await lease._settlement


def test_guarded_trace_retains_names_and_model_but_drops_payloads() -> None:
    with Session() as session:
        with _guarded_trace():
            session.emit(
                EventType.TOOL_CALL,
                {
                    "tool_name": "search",
                    "model": "model-x",
                    "content": "secret",
                    "arguments": {"query": "secret"},
                    "output": "secret result",
                },
            )
        [event] = session.events.query(event_type=EventType.TOOL_CALL)
    assert event["payload"] == {"tool_name": "search", "model": "model-x", "guarded": True}


def test_probe_pid_treats_a_process_of_another_user_as_alive(monkeypatch):
    """EPERM / ERROR_ACCESS_DENIED mean the process exists: never reclaim it."""
    import os

    from lazybridge import _process_owner

    if os.name == "nt":
        import ctypes

        class _Kernel32:
            def OpenProcess(self, access, inherit, pid):
                return 0

        monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: _Kernel32())
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 5)
        assert _process_owner.probe_pid(424242) is True
        monkeypatch.setattr(ctypes, "get_last_error", lambda: 87)
        assert _process_owner.probe_pid(424242) is False
    else:

        def _deny(pid, sig):
            raise PermissionError

        monkeypatch.setattr(os, "kill", _deny)
        assert _process_owner.probe_pid(424242) is True
