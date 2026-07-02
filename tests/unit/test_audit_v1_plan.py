"""Regression tests for the v1-stabilization Plan/Replan/checkpoint audit fixes.

* F5 — a cancelled run (e.g. a ``plan.stream()`` consumer disconnecting)
  left the checkpoint key stuck in ``claimed``/``running`` under a dead
  ``run_uid``: every later ``on_concurrent="fail"`` run raised
  ``ConcurrentPlanRunError`` until the key was manually cleared.  The
  engine now writes a terminal checkpoint (``cancelled`` on cancellation,
  ``done`` on conclude, ``failed`` on unexpected exceptions) and
  ``_claim_checkpoint`` treats ``cancelled`` as claimable/adoptable.
* F6 — ``to_dict`` dropped ``Step.output`` / ``Step.input``: every
  structured step silently degraded to ``output=str`` on reload and
  ``routes_by`` plans failed to recompile.
* F7 — routing (``routes=`` / ``routes_by=``) into a ``parallel=True``
  step compiled cleanly but silently lost the ``after_branches`` rejoin
  jump at runtime; the compiler now rejects it.
* Replan — same cancellation poisoning as F5, fixed the same way.
* ``EventLog.flush()`` after ``close()`` stalled for the full timeout.
* ``EncryptedStoreAdapter`` lacked the context-manager protocol and bulk
  reads didn't name the offending key on mixed plaintext rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from lazybridge.engines.plan import Plan, PlanCompileError, Step
from lazybridge.envelope import Envelope
from lazybridge.store import Store


def _run_plan(plan: Plan, task: str = "go") -> Envelope:
    async def _go():
        return await plan.run(
            Envelope.from_task(task),
            tools=[],
            output_type=str,
            memory=None,
            session=None,
        )

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# F5 — terminal checkpoint on cancellation / conclude / unexpected error
# ---------------------------------------------------------------------------


def _make_cancellable_plan(store: Store, key: str, *, resume: bool = False) -> tuple[Plan, dict]:
    calls: dict[str, int] = {"a": 0, "b": 0}

    def step_a(task: str) -> str:
        calls["a"] += 1
        return "a-done"

    async def step_b(task: str) -> str:
        calls["b"] += 1
        await asyncio.sleep(5)
        return "b-done"

    plan = Plan(
        Step(step_a, name="a", writes="out_a"),
        Step(step_b, name="b"),
        store=store,
        checkpoint_key=key,
        resume=resume,
    )
    return plan, calls


def test_cancelled_run_leaves_terminal_checkpoint():
    store = Store()
    plan, _ = _make_cancellable_plan(store, "k1")

    async def _go():
        task = asyncio.create_task(
            plan.run(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
        )
        await asyncio.sleep(0.1)  # step "a" done, step "b" sleeping
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(_go())
    snap = store.read("k1")
    assert snap["status"] == "cancelled"
    assert snap["next_step"] == "b"
    assert snap["completed_steps"] == ["a"]
    assert snap["kv"] == {"out_a": "a-done"}


def test_cancelled_key_is_claimable_by_a_fresh_run():
    store = Store()
    plan, _ = _make_cancellable_plan(store, "k2")

    async def _cancel_run():
        task = asyncio.create_task(
            plan.run(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(_cancel_run())
    assert store.read("k2")["status"] == "cancelled"

    # A fresh run (resume=False) claims over the cancelled key instead of
    # raising ConcurrentPlanRunError forever.
    fast = Plan(
        Step(lambda task: "x", name="a", writes="out_a"),
        Step(lambda task: "y", name="b"),
        store=store,
        checkpoint_key="k2",
    )
    result = _run_plan(fast)
    assert result.ok
    assert store.read("k2")["status"] == "done"


def test_cancelled_key_resumes_from_next_step():
    store = Store()
    plan, _ = _make_cancellable_plan(store, "k3")

    async def _cancel_run():
        task = asyncio.create_task(
            plan.run(Envelope.from_task("go"), tools=[], output_type=str, memory=None, session=None)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(_cancel_run())

    resumed, calls = _make_cancellable_plan(store, "k3", resume=True)
    # Swap step b's target for a fast one so the resume completes.
    resumed.steps[1] = Step(lambda task: "b-fast", name="b")
    result = _run_plan(resumed)
    assert result.ok
    # Step "a" completed in the first run — resume must not re-run it.
    assert calls["a"] == 0
    assert store.read("k3")["status"] == "done"


def test_stream_early_close_leaves_terminal_checkpoint():
    store = Store()
    plan, _ = _make_cancellable_plan(store, "k4")

    async def _go():
        agen = plan.stream(
            Envelope.from_task("go"),
            tools=[],
            output_type=str,
            memory=None,
            session=None,
        ).__aiter__()
        pending = asyncio.create_task(agen.__anext__())
        await asyncio.sleep(0.1)  # plan is inside step "b"'s sleep
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=5)
        await agen.aclose()

    asyncio.run(_go())
    snap = store.read("k4")
    assert snap is not None and snap["status"] == "cancelled"


def test_conclude_marks_checkpoint_done():
    from lazybridge.signals import ConcludeSignal, conclude

    store = Store()

    def fin(task: str) -> str:
        return conclude("early answer")

    plan = Plan(
        Step(lambda task: "a", name="a"),
        Step(fin, name="fin"),
        store=store,
        checkpoint_key="k5",
    )
    with pytest.raises(ConcludeSignal):
        _run_plan(plan)
    snap = store.read("k5")
    assert snap["status"] == "done"
    assert snap["next_step"] is None


def test_unexpected_exception_marks_checkpoint_failed():
    from lazybridge.engines.plan import PlanRuntimeError

    store = Store()

    def boom(env) -> bool:
        raise RuntimeError("predicate bug")

    plan = Plan(
        Step(lambda task: "a", name="a", routes={"b": boom}),
        Step(lambda task: "b", name="b"),
        store=store,
        checkpoint_key="k6",
    )
    with pytest.raises(PlanRuntimeError):
        _run_plan(plan)
    snap = store.read("k6")
    assert snap["status"] == "failed"


# ---------------------------------------------------------------------------
# F6 — Step.output / Step.input survive serialization
# ---------------------------------------------------------------------------


class _Triage(BaseModel):
    severity: Literal["urgent", "normal"] = "urgent"


def test_step_output_and_input_round_trip():
    def classify(task: str) -> str:
        return "x"

    plan = Plan(
        Step(classify, name="classify", output=_Triage),
        Step(lambda task: "y", name="handle", input=_Triage),
    )
    data = plan.to_dict()
    assert data["version"] == 2
    step_dicts = {s["name"]: s for s in data["steps"]}
    assert step_dicts["classify"]["output"] == "_Triage"
    assert step_dicts["handle"]["input"] == "_Triage"

    plan2 = Plan.from_dict(
        data,
        registry={
            "classify": classify,
            "<lambda>": lambda task: "y",
            "type:_Triage": _Triage,
        },
    )
    steps = {s.name: s for s in plan2.steps}
    assert steps["classify"].output is _Triage
    assert steps["handle"].input is _Triage


def test_missing_type_registry_entry_fails_loud():
    plan = Plan(Step(lambda task: "x", name="s", output=_Triage))
    data = plan.to_dict()
    with pytest.raises(KeyError, match="type:_Triage"):
        Plan.from_dict(data, registry={"<lambda>": lambda task: "x"})


def test_v1_payload_without_output_still_loads():
    # Old (v1) payloads have no output/input keys — they default to str/Any.
    data = {
        "version": 1,
        "max_iterations": 100,
        "steps": [{"name": "s", "target": {"kind": "tool", "name": "t"}, "task": None, "parallel": False}],
    }
    plan = Plan.from_dict(data)
    assert plan.steps[0].output is str
    assert plan.steps[0].input is Any


# ---------------------------------------------------------------------------
# F7 — routing into a parallel band is a compile error
# ---------------------------------------------------------------------------


def test_routes_to_parallel_step_is_compile_error():
    plan = Plan(
        Step(lambda task: "t", name="triage", routes={"branch_a": lambda env: True}),
        Step(lambda task: "a", name="branch_a", parallel=True),
        Step(lambda task: "b", name="branch_b", parallel=True),
    )
    with pytest.raises(PlanCompileError, match="parallel band"):
        plan._validate({})


def test_routes_by_to_parallel_step_is_compile_error():
    class _Route(BaseModel):
        kind: Literal["branch_a", "done"] = "done"

    plan = Plan(
        Step(lambda task: "t", name="triage", output=_Route, routes_by="kind"),
        Step(lambda task: "a", name="branch_a", parallel=True),
        Step(lambda task: "b", name="branch_b", parallel=True),
        Step(lambda task: "d", name="done"),
    )
    with pytest.raises(PlanCompileError, match="parallel"):
        plan._validate({})


def test_routes_to_sequential_step_still_compiles():
    plan = Plan(
        Step(lambda task: "t", name="triage", routes={"handle": lambda env: True}),
        Step(lambda task: "h", name="handle"),
    )
    plan._validate({})  # must not raise


# ---------------------------------------------------------------------------
# Replan — cancellation leaves a terminal checkpoint
# ---------------------------------------------------------------------------


def test_replan_cancellation_leaves_terminal_checkpoint():
    from lazybridge.engines.replan import PlanRound, ReplanEngine
    from lazybridge.tools import Tool

    store = Store()

    async def planner(task: str) -> PlanRound:
        await asyncio.sleep(5)
        return PlanRound(reasoning="r", tasks=[], done=True, final_answer="x")

    engine = ReplanEngine(store=store, checkpoint_key="rk1")
    tools = [Tool(planner, name="planner")]

    async def _go():
        task = asyncio.create_task(
            engine.run(Envelope.from_task("go"), tools=tools, output_type=str, memory=None, session=None)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    asyncio.run(_go())
    snap = store.read("rk1")
    assert snap["status"] == "cancelled"

    # A fresh run claims over the cancelled key.
    async def quick_planner(task: str) -> PlanRound:
        return PlanRound(reasoning="r", tasks=[], done=True, final_answer="answer")

    engine2 = ReplanEngine(store=store, checkpoint_key="rk1")

    async def _rerun():
        return await engine2.run(
            Envelope.from_task("go"),
            tools=[Tool(quick_planner, name="planner")],
            output_type=str,
            memory=None,
            session=None,
        )

    result = asyncio.run(_rerun())
    assert result.ok
    assert result.payload == "answer"
    assert store.read("rk1")["status"] == "done"


# ---------------------------------------------------------------------------
# EventLog.flush() after close() is a no-op
# ---------------------------------------------------------------------------


def test_flush_after_close_returns_immediately():
    from lazybridge.session import EventLog

    log = EventLog("flush-test", batched=True)
    log.record("AGENT_START", {"agent_name": "x"})
    log.close()
    t0 = time.monotonic()
    log.flush(timeout=5.0)
    assert time.monotonic() - t0 < 1.0


# ---------------------------------------------------------------------------
# EncryptedStoreAdapter — context manager + keyed decrypt errors
# ---------------------------------------------------------------------------


def test_encrypted_adapter_context_manager():
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    from lazybridge.store.encryption import EncryptedStoreAdapter

    with EncryptedStoreAdapter(Store(), key=Fernet.generate_key()) as s:
        s.write("k", {"v": 1})
        assert s.read("k") == {"v": 1}


def test_encrypted_adapter_bulk_read_names_offending_key():
    pytest.importorskip("cryptography")
    from cryptography.fernet import Fernet

    from lazybridge.store.encryption import EncryptedStoreAdapter

    inner = Store()
    inner.write("plain-row", "not encrypted")
    adapter = EncryptedStoreAdapter(inner, key=Fernet.generate_key())
    adapter.write("enc-row", "secret")
    with pytest.raises(ValueError, match="plain-row"):
        adapter.read_all()
    with pytest.raises(ValueError, match="plain-row"):
        adapter.items()
