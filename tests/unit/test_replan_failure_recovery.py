"""ReplanEngine failure-injection & resume-recovery contract.

The replan loop's durability promise: a *transient* failure (a worker tool,
a parallel branch, or the planner itself) fails the run cleanly with a
checkpoint at the failed round, and ``resume=True`` re-runs that round and
recovers — WITHOUT re-executing rounds that already completed.  Two
non-recovery invariants round it out: an error-Envelope worker must surface
as an error (never a phantom-empty success the planner would replan from),
and a legitimately-empty result must flow to the planner so it can replan.

Existing replan coverage exercises cancellation → terminal checkpoint and a
plain done-resume; this pins the failure→resume→recover path (the property a
future refactor is most likely to break silently).

Style mirrors ``test_audit_v1_plan.py``: an in-memory ``Store()`` shared
across two engines, planner as an ``async def ... -> PlanRound`` tool, driven
via ``engine.run(...)`` under ``asyncio.run``.
"""

from __future__ import annotations

import asyncio

from lazybridge import Store
from lazybridge.engines.replan import PlanRound, ReplanEngine, ReplanTask
from lazybridge.envelope import Envelope
from lazybridge.tools import Tool


def _run(engine: ReplanEngine, tools: list[Tool], task: str = "go") -> Envelope:
    return asyncio.run(
        engine.run(
            Envelope.from_task(task),
            tools=tools,
            output_type=str,
            memory=None,
            session=None,
        )
    )


def _engine(store: Store, *, resume: bool, key: str = "rk", max_rounds: int = 20) -> ReplanEngine:
    return ReplanEngine(store=store, checkpoint_key=key, resume=resume, max_rounds=max_rounds)


# ---------------------------------------------------------------------------
# 1. Transient worker failure → resume re-runs the round and recovers.
# ---------------------------------------------------------------------------
def test_transient_worker_failure_recovers_on_resume():
    calls = {"n": 0}

    def source(q: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient source outage")
        return "SOURCE_OK"

    async def planner(task: str) -> PlanRound:
        if "SOURCE_OK" in task:
            return PlanRound(reasoning="have data", tasks=[], done=True, final_answer="got-data")
        return PlanRound(
            reasoning="fetch",
            tasks=[ReplanTask(tool="source", kwargs={"q": "x"}, parallel=False)],
            done=False,
        )

    store = Store()
    tools = [Tool(planner, name="planner"), Tool(source, name="source")]

    r1 = _run(_engine(store, resume=True), tools)
    assert not r1.ok, "run 1 must fail on the transient source error"
    assert store.read("rk")["status"] == "failed"

    r2 = _run(_engine(store, resume=True), tools)
    assert r2.ok and r2.payload == "got-data", f"resume did not recover: {r2!r}"
    assert calls["n"] == 2, "source should have been retried exactly once on resume"


# ---------------------------------------------------------------------------
# 2. An error-Envelope worker surfaces as an error (no phantom-empty success).
# ---------------------------------------------------------------------------
def test_error_envelope_worker_is_surfaced_not_phantom_empty():
    def source(q: str) -> Envelope:
        return Envelope.error_envelope(ValueError("malformed upstream payload"))

    async def planner(task: str) -> PlanRound:
        return PlanRound(
            reasoning="fetch",
            tasks=[ReplanTask(tool="source", kwargs={"q": "x"}, parallel=False)],
            done=False,
        )

    store = Store()
    tools = [Tool(planner, name="planner"), Tool(source, name="source")]

    r = _run(_engine(store, resume=False, key="e2"), tools)
    assert not r.ok, "an error-Envelope worker must fail the run, not read as empty success"
    assert r.error is not None and "error" in r.error.message.lower()


# ---------------------------------------------------------------------------
# 3. A legitimately-empty worker result flows to the planner, which replans.
# ---------------------------------------------------------------------------
def test_empty_worker_result_flows_to_planner():
    calls = {"n": 0}

    def source(q: str) -> str:
        calls["n"] += 1
        return "" if calls["n"] == 1 else "SOURCE_OK"

    async def planner(task: str) -> PlanRound:
        if "SOURCE_OK" in task:
            return PlanRound(reasoning="ok", tasks=[], done=True, final_answer="eventually")
        return PlanRound(
            reasoning="fetch",
            tasks=[ReplanTask(tool="source", kwargs={"q": "x"}, parallel=False)],
            done=False,
        )

    store = Store()
    tools = [Tool(planner, name="planner"), Tool(source, name="source")]

    r = _run(_engine(store, resume=False, key="e3", max_rounds=10), tools)
    assert r.ok and r.payload == "eventually", f"empty result mishandled: {r!r}"
    assert calls["n"] >= 2, "planner should have replanned past the empty result"


# ---------------------------------------------------------------------------
# 4. A parallel-band branch failure recovers on resume.
# ---------------------------------------------------------------------------
def test_parallel_band_failure_recovers_on_resume():
    calls = {"n": 0}

    def flaky(q: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("parallel branch outage")
        return "SOURCE_OK"

    def other(q: str) -> str:
        return "sibling-ok"

    async def planner(task: str) -> PlanRound:
        if "SOURCE_OK" in task:
            return PlanRound(reasoning="ok", tasks=[], done=True, final_answer="par-done")
        return PlanRound(
            reasoning="fan-out",
            tasks=[
                ReplanTask(tool="flaky", kwargs={"q": "x"}, parallel=True),
                ReplanTask(tool="other", kwargs={"q": "y"}, parallel=True),
            ],
            done=False,
        )

    store = Store()
    tools = [Tool(planner, name="planner"), Tool(flaky, name="flaky"), Tool(other, name="other")]

    r1 = _run(_engine(store, resume=True, key="e4"), tools)
    assert not r1.ok
    r2 = _run(_engine(store, resume=True, key="e4"), tools)
    assert r2.ok and r2.payload == "par-done", f"parallel-band recovery failed: {r2!r}"


# ---------------------------------------------------------------------------
# 5. A transient planner failure recovers on resume.
# ---------------------------------------------------------------------------
def test_transient_planner_failure_recovers_on_resume():
    pcalls = {"n": 0}

    async def planner(task: str) -> PlanRound:
        pcalls["n"] += 1
        if pcalls["n"] == 1:
            raise RuntimeError("planner LLM hiccup")
        return PlanRound(reasoning="done", tasks=[], done=True, final_answer="planner-ok")

    store = Store()
    tools = [Tool(planner, name="planner")]

    r1 = _run(_engine(store, resume=True, key="e5"), tools)
    assert not r1.ok, "run 1 must fail on the planner error"
    assert store.read("e5")["status"] == "failed"

    r2 = _run(_engine(store, resume=True, key="e5"), tools)
    assert r2.ok and r2.payload == "planner-ok", f"planner-failure recovery failed: {r2!r}"


# ---------------------------------------------------------------------------
# 6. A failure in a LATER round recovers without re-running completed rounds.
# ---------------------------------------------------------------------------
def test_later_round_failure_does_not_rerun_completed_rounds():
    fetch_calls = {"n": 0}
    refine_calls = {"n": 0}

    def fetch(q: str) -> str:
        fetch_calls["n"] += 1
        return "RAW"

    def refine(q: str) -> str:
        refine_calls["n"] += 1
        if refine_calls["n"] == 1:
            raise RuntimeError("refine outage (round 1)")
        return "REFINED"

    async def planner(task: str) -> PlanRound:
        if "REFINED" in task:
            return PlanRound(reasoning="ok", tasks=[], done=True, final_answer="two-round")
        if "RAW" in task:  # round 0 already completed → do round 1
            return PlanRound(
                reasoning="refine",
                tasks=[ReplanTask(tool="refine", kwargs={"q": "x"}, parallel=False)],
                done=False,
            )
        return PlanRound(
            reasoning="fetch",
            tasks=[ReplanTask(tool="fetch", kwargs={"q": "x"}, parallel=False)],
            done=False,
        )

    store = Store()
    tools = [Tool(planner, name="planner"), Tool(fetch, name="fetch"), Tool(refine, name="refine")]

    r1 = _run(_engine(store, resume=True, key="e6"), tools)
    assert not r1.ok, "run 1 must fail in round 1 (refine)"

    r2 = _run(_engine(store, resume=True, key="e6"), tools)
    assert r2.ok and r2.payload == "two-round", f"later-round recovery failed: {r2!r}"
    # The key invariant: round 0's fetch is NOT re-run on resume — its output
    # persisted in the checkpoint history, so only the failed round replays.
    assert fetch_calls["n"] == 1, f"round-0 task re-ran on resume (fetch_calls={fetch_calls['n']})"
    assert refine_calls["n"] == 2, f"refine should retry exactly once (refine_calls={refine_calls['n']})"
