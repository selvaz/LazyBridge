"""Tests for ``Plan.run_many`` / ``Plan.arun_many``.

These cover the boilerplate-elimination contract: callers shouldn't
need ``ThreadPoolExecutor`` + ``asyncio.run`` to fan-out a Plan over
multiple inputs.  ``run_many`` owns that mechanic.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from lazybridge import Plan, Step, Store
from lazybridge.envelope import Envelope
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Sync surface
# ---------------------------------------------------------------------------


def test_run_many_returns_list_in_input_order() -> None:
    """The result list must match the input order one-to-one."""

    def respond(env: Envelope) -> str:
        return f"processed:{env.task}"

    agent = MockAgent(respond, name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})

    inputs = ["alpha", "beta", "gamma", "delta"]
    results = plan.run_many(inputs)

    assert len(results) == len(inputs)
    for input_value, result in zip(inputs, results, strict=True):
        assert result.ok
        assert result.payload == f"processed:{input_value}"


def test_run_many_accepts_envelopes_as_well_as_strings() -> None:
    """Pre-built envelopes can be passed alongside plain strings.

    Use ``Envelope.from_task`` (or pass the string directly) so the
    payload carries the task — same convention as the rest of the
    framework.
    """

    def respond(env: Envelope) -> str:
        return env.task or ""

    agent = MockAgent(respond, name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})

    results = plan.run_many([Envelope.from_task("A"), "B", Envelope.from_task("C")])
    payloads = [r.payload for r in results]
    assert payloads == ["A", "B", "C"]


def test_run_many_per_task_errors_appear_as_error_envelopes() -> None:
    """A task that triggers an error inside the Plan must produce an
    error envelope in its slot — never raise out of run_many."""

    def respond(env: Envelope) -> str:
        if env.task == "boom":
            raise RuntimeError("simulated failure")
        return "ok"

    agent = MockAgent(respond, name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})

    results = plan.run_many(["ok", "boom", "ok"])
    assert results[0].ok
    assert not results[1].ok
    assert results[1].error is not None
    assert "simulated failure" in results[1].error.message
    assert results[2].ok


def test_run_many_concurrency_cap_is_respected() -> None:
    """``concurrency=N`` caps the simultaneous in-flight runs.

    We measure peak concurrency by recording how many tasks observe
    the gate as taken at any moment.  With ``concurrency=2`` and 6
    tasks each holding the slot for ~50ms, peak should be ≤ 2.
    """
    in_flight = {"current": 0, "peak": 0}

    async def respond(env: Envelope) -> str:
        in_flight["current"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
        await asyncio.sleep(0.05)
        in_flight["current"] -= 1
        return env.task or ""

    agent = MockAgent(respond, name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})

    plan.run_many([f"t{i}" for i in range(6)], concurrency=2)
    assert in_flight["peak"] <= 2, f"peak concurrency exceeded cap: {in_flight['peak']}"


def test_run_many_with_fork_keeps_each_run_isolated() -> None:
    """``Plan(on_concurrent='fork')`` + ``run_many`` must give every
    fan-out run its own keyspace — no cross-contamination of writes."""

    def respond(env: Envelope) -> str:
        return f"out-{env.task}"

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "fork.sqlite"))
        agent = MockAgent(respond, name="agent")
        plan = Plan(
            Step(agent, name="step", writes="result"),
            store=store,
            checkpoint_key="run_many_fork",
            on_concurrent="fork",
        )
        plan._validate({})

        n = 5
        results = plan.run_many([f"task-{i}" for i in range(n)])
        assert all(r.ok for r in results)

        # Each forked run claimed its own ``run_many_fork:<uid>`` key.
        forked_keys = [k for k in store if k.startswith("run_many_fork:")]
        assert len(forked_keys) >= n


def test_run_many_empty_input_returns_empty_list() -> None:
    agent = MockAgent("ok", name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})
    assert plan.run_many([]) == []


# ---------------------------------------------------------------------------
# Async surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_many_works_inside_event_loop() -> None:
    """``arun_many`` is the async counterpart — usable from inside an
    already-running event loop without the sync-bridge overhead."""

    def respond(env: Envelope) -> str:
        return env.task or ""

    agent = MockAgent(respond, name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})

    results = await plan.arun_many(["a", "b", "c"])
    assert [r.payload for r in results] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Sync façade nesting — calling run_many from inside another sync caller
# ---------------------------------------------------------------------------


def test_run_many_works_when_called_from_sync_python_code() -> None:
    """The sync-bridge propagates the caller's contextvars, so ``run_many``
    can be invoked from any sync entry point (CLI scripts, pytest tests,
    background workers) without manual ``asyncio.run`` setup."""

    def respond(env: Envelope) -> str:
        return f"got:{env.task}"

    agent = MockAgent(respond, name="agent")
    plan = Plan(Step(agent, name="step"))
    plan._validate({})

    # No event loop, no asyncio.run, no ThreadPoolExecutor.
    results = plan.run_many(["x", "y", "z"])
    assert [r.payload for r in results] == ["got:x", "got:y", "got:z"]
