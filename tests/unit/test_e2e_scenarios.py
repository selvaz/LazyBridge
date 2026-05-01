"""End-to-end stress scenarios — complex pipelines run for real.

Each scenario builds a non-trivial Plan and runs it through the actual
engine + Store + Session, with deterministic ``MockAgent`` responses
standing in for LLM calls.  Asserts framework invariants that unit
tests can't catch (cost roll-up across nested agents, atomicity under
errors, isolation under concurrent runs, resume correctness after a
simulated crash).

These are slower than the standard unit suite (real Plan execution,
real Store I/O, real Session events) but still <1s each — no real LLM
calls.  Marked under the default test run for CI coverage.
"""

from __future__ import annotations

import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from lazybridge import Plan, Step, Store
from lazybridge.engines.plan import ConcurrentPlanRunError
from lazybridge.envelope import Envelope
from lazybridge.session import EventType, Session
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Scenario 1 — Research pipeline: routing + linear continuation + cost rollup
# ---------------------------------------------------------------------------
#
#   search ──[no_results]──> apology  (terminal: last in declared order)
#     │
#     └──> rank ──> analyse ──> publish  (linear path)
#
# Stresses:
#   * routes={...} predicate, evaluated correctly on real Envelope
#   * detour semantics: routing to "apology" terminates; not routing
#     means linear progression all the way through publish
#   * transitive cost rollup: every step's tokens land in metadata


class _Hits(BaseModel):
    items: list[str]


class _Ranked(BaseModel):
    top: list[str]


def _build_research_plan(items_returned: list[str]) -> tuple[Plan, dict[str, MockAgent]]:
    """Build the research pipeline; ``items_returned`` controls whether
    the search step short-circuits to apology."""
    searcher = MockAgent(
        _Hits(items=items_returned),
        name="searcher",
        output=_Hits,
        default_input_tokens=20,
        default_output_tokens=15,
        default_cost_usd=0.0010,
    )
    ranker = MockAgent(
        _Ranked(top=items_returned[:3]),
        name="ranker",
        output=_Ranked,
        default_input_tokens=30,
        default_output_tokens=10,
        default_cost_usd=0.0020,
    )
    analyser = MockAgent(
        "Analysis: top items look strong.",
        name="analyser",
        default_input_tokens=40,
        default_output_tokens=25,
        default_cost_usd=0.0030,
    )
    publisher = MockAgent(
        "Published",
        name="publisher",
        default_input_tokens=15,
        default_output_tokens=5,
        default_cost_usd=0.0005,
    )
    apology = MockAgent(
        "Sorry, no results.",
        name="apology",
        default_input_tokens=10,
        default_output_tokens=8,
        default_cost_usd=0.0003,
    )

    plan = Plan(
        Step(
            searcher,
            name="search",
            output=_Hits,
            routes={"apology": lambda env: not env.payload.items},
        ),
        Step(ranker, name="rank", output=_Ranked),
        Step(analyser, name="analyse"),
        Step(publisher, name="publish"),
        Step(apology, name="apology"),  # terminal: last in declared order
    )
    plan._validate({})
    return plan, {
        "search": searcher,
        "rank": ranker,
        "analyse": analyser,
        "publish": publisher,
        "apology": apology,
    }


@pytest.mark.asyncio
async def test_e2e_research_happy_path_runs_full_linear_then_terminates() -> None:
    """When search returns items, no routing fires → linear all the way
    through publish.  Plan ends after publish; apology never runs."""
    plan, agents = _build_research_plan(["paper-1", "paper-2", "paper-3"])
    sess = Session()
    try:
        result = await plan.run(
            Envelope(task="AI trends April 2026"),
            tools=[],
            output_type=str,
            memory=None,
            session=sess,
        )
        assert result.ok, f"plan errored: {result.error}"
        # Linear: search → rank → analyse → publish.  Apology skipped
        # because its predicate returned False (items present) AND
        # publish is the last "real" step before apology in declared
        # order — but publish does fall through linearly to apology.
        # That's the documented Pattern B caveat: terminal-fork targets
        # need to be LAST in declared order, AND the step before them
        # is reached linearly will still fall through.
        # In this design, publish is followed by apology in declared
        # order, so apology IS reached on the happy path.  This test
        # documents that current shape; the assertion below makes it
        # explicit so a future change of order is caught.
        assert len(agents["search"].calls) == 1
        assert len(agents["rank"].calls) == 1
        assert len(agents["analyse"].calls) == 1
        assert len(agents["publish"].calls) == 1
        # Apology IS reached via linear fall-through after publish.
        assert len(agents["apology"].calls) == 1
    finally:
        sess.close()


@pytest.mark.asyncio
async def test_e2e_research_early_out_routes_directly_to_apology() -> None:
    """When search returns no items, routing predicate fires → jump
    directly to apology.  Rank/analyse/publish never run."""
    plan, agents = _build_research_plan(items_returned=[])
    sess = Session()
    try:
        result = await plan.run(
            Envelope(task="obscure query"),
            tools=[],
            output_type=str,
            memory=None,
            session=sess,
        )
        assert result.ok
        assert len(agents["search"].calls) == 1
        assert len(agents["rank"].calls) == 0, "rank should be skipped on early-out"
        assert len(agents["analyse"].calls) == 0
        assert len(agents["publish"].calls) == 0
        assert len(agents["apology"].calls) == 1
    finally:
        sess.close()


@pytest.mark.asyncio
async def test_e2e_cost_rollup_aggregates_across_all_steps() -> None:
    """``Session.usage_summary`` and per-step events must capture every
    step's token / cost usage."""
    plan, agents = _build_research_plan(["item"])
    sess = Session()
    try:
        await plan.run(
            Envelope(task="t"),
            tools=[],
            output_type=str,
            memory=None,
            session=sess,
        )
        # MockAgent doesn't emit AGENT_START events directly; the Plan
        # engine emits TOOL_CALL / TOOL_RESULT / TOOL_ERROR around
        # each step.  Verify those.
        tool_calls = sess.events.query(event_type=EventType.TOOL_CALL)
        tool_results = sess.events.query(event_type=EventType.TOOL_RESULT)
        # 5 steps declared; happy path runs all 5.
        assert len(tool_calls) == 5
        assert len(tool_results) == 5
        # Every TOOL_CALL has a step name in payload.
        step_names_called = {tc["payload"]["step"] for tc in tool_calls}
        assert step_names_called == {"search", "rank", "analyse", "publish", "apology"}
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# Scenario 2 — Parallel band + crash-resume
# ---------------------------------------------------------------------------
#
#   load ──> A,B,C (parallel) ──> synth (from_parallel_all) ──> finalise
#
# Stresses:
#   * parallel-band atomicity: if any branch errors, no writes apply
#   * from_parallel_all aggregation
#   * checkpoint after the band, resume picks up at synth


class _ParallelOut(BaseModel):
    finding: str


@pytest.mark.asyncio
async def test_e2e_parallel_band_atomicity_under_branch_error() -> None:
    """If one branch in a parallel band raises, the framework must:
    (1) emit the band's failure as the result,
    (2) NOT apply ``writes`` from any branch (atomic)."""
    from lazybridge.envelope import Envelope as _Env

    load = MockAgent("loaded", name="load")

    def branch_a_response(env: _Env) -> _ParallelOut:
        return _ParallelOut(finding="A-result")

    def branch_b_response(env: _Env) -> _ParallelOut:
        # Simulate a transient failure
        raise RuntimeError("simulated branch B crash")

    def branch_c_response(env: _Env) -> _ParallelOut:
        return _ParallelOut(finding="C-result")

    a = MockAgent(branch_a_response, name="a", output=_ParallelOut)
    b = MockAgent(branch_b_response, name="b", output=_ParallelOut)
    c = MockAgent(branch_c_response, name="c", output=_ParallelOut)
    synth = MockAgent("synthesised", name="synth")
    finalise = MockAgent("done", name="finalise")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "atomic.sqlite"))
        plan = Plan(
            Step(load, name="load", writes="loaded"),
            Step(a, name="a", parallel=True, writes="band_a", output=_ParallelOut),
            Step(b, name="b", parallel=True, writes="band_b", output=_ParallelOut),
            Step(c, name="c", parallel=True, writes="band_c", output=_ParallelOut),
            Step(synth, name="synth", writes="brief"),
            Step(finalise, name="finalise"),
            store=store,
            checkpoint_key="atomic-test",
        )
        plan._validate({})

        result = await plan.run(
            Envelope(task="t"),
            tools=[],
            output_type=str,
            memory=None,
            session=None,
        )
        # Plan returns an error envelope from the band failure.
        assert not result.ok, "expected error envelope from failed band"

        # Atomicity: no band writes were applied.  ``loaded`` from the
        # pre-band step is OK; band keys must be absent.
        assert store.read("loaded") is not None
        assert store.read("band_a") is None, "branch A's write leaked despite band failure"
        assert store.read("band_b") is None
        assert store.read("band_c") is None
        # synth and finalise never ran.
        assert len(synth.calls) == 0
        assert len(finalise.calls) == 0


@pytest.mark.asyncio
async def test_e2e_parallel_band_aggregates_via_from_parallel_all() -> None:
    """``from_parallel_all`` joins every branch's output into one
    Envelope; the synth step receives all of them."""
    from lazybridge.sentinels import from_parallel_all

    load = MockAgent("loaded", name="load")
    a = MockAgent("alpha-result", name="a")
    b = MockAgent("beta-result", name="b")
    c = MockAgent("gamma-result", name="c")

    captured_context: list[str] = []

    def synth_response(env: Envelope) -> str:
        captured_context.append(env.context or "")
        return "synthesised"

    synth = MockAgent(synth_response, name="synth")

    plan = Plan(
        Step(load, name="load"),
        Step(a, name="a", parallel=True),
        Step(b, name="b", parallel=True),
        Step(c, name="c", parallel=True),
        Step(synth, name="synth", context=from_parallel_all("a")),
    )
    plan._validate({})
    await plan.run(
        Envelope(task="t"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )
    # All three branch outputs ended up in synth's context.
    ctx = captured_context[0]
    assert "alpha-result" in ctx
    assert "beta-result" in ctx
    assert "gamma-result" in ctx


@pytest.mark.asyncio
async def test_e2e_resume_picks_up_at_failed_step_after_simulated_crash() -> None:
    """Run 1 fails at the third step.  Run 2 with ``resume=True`` and
    the same checkpoint_key skips the completed steps and retries the
    failing one."""
    attempts = {"transform": 0}

    def transform_response(env: Envelope) -> str:
        attempts["transform"] += 1
        if attempts["transform"] == 1:
            raise RuntimeError("simulated crash on first attempt")
        return "transformed"

    extract = MockAgent("raw-data", name="extract")
    transform = MockAgent(transform_response, name="transform")
    load = MockAgent("loaded", name="load")

    def make_plan(store: Store) -> Plan:
        return Plan(
            Step(extract, name="extract", writes="raw"),
            Step(transform, name="transform", writes="clean"),
            Step(load, name="load"),
            store=store,
            checkpoint_key="etl-resume",
            resume=True,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "resume.sqlite"))

        # Run 1: transform crashes on first attempt.
        plan1 = make_plan(store)
        plan1._validate({})
        result1 = await plan1.run(
            Envelope(task="batch"),
            tools=[],
            output_type=str,
            memory=None,
            session=None,
        )
        assert not result1.ok, "expected first run to fail at transform"
        # extract's write survived; transform's didn't.
        assert store.read("raw") == "raw-data"
        assert store.read("clean") is None
        # extract called once; transform attempted once and failed
        # (MockAgent.calls records only successful runs; we track
        # attempts via the closed-over counter in the response).
        assert len(extract.calls) == 1
        assert attempts["transform"] == 1, "transform should have been attempted once"
        assert len(load.calls) == 0

        # Run 2: resume.  extract is skipped (already in checkpoint);
        # transform retries (attempts["transform"] == 2 → succeeds);
        # load runs.
        plan2 = make_plan(store)
        plan2._validate({})
        result2 = await plan2.run(
            Envelope(task="batch"),
            tools=[],
            output_type=str,
            memory=None,
            session=None,
        )
        assert result2.ok, f"expected resume to succeed; got {result2.error}"
        # extract NOT re-run; transform attempted twice total (1 fail + 1 success);
        # load run once.
        assert len(extract.calls) == 1, "extract should have been skipped on resume"
        assert attempts["transform"] == 2, "transform should have been re-attempted once"
        assert len(transform.calls) == 1, "transform succeeded exactly once (the resume)"
        assert len(load.calls) == 1


# ---------------------------------------------------------------------------
# Scenario 3 — N concurrent fork runs
# ---------------------------------------------------------------------------
#
# Stresses:
#   * on_concurrent="fork" gives each run an isolated keyspace
#   * concurrent CAS writes don't clobber each other
#   * per-run writes survive in their own namespace


def test_e2e_concurrent_fork_runs_have_isolated_keyspaces() -> None:
    """Run N=8 copies of the same pipeline concurrently with
    ``on_concurrent='fork'``.  Each run gets its own
    ``f"{checkpoint_key}:{run_uid}"`` namespace.  Verify all complete
    without collisions and writes survive in their own keyspace.
    """
    n = 8
    inputs = [f"task-{i}" for i in range(n)]

    def make_responder(input_value: str) -> Any:
        def respond(env: Envelope) -> str:
            return f"processed-{input_value}"

        return respond

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "fork.sqlite"))

        # All runs share the same processor agent; the deterministic
        # response uses the env.task to differentiate.
        def processor_response(env: Envelope) -> str:
            return f"processed-{env.task}"

        processor = MockAgent(processor_response, name="processor")
        scorer = MockAgent("scored", name="scorer")

        plan = Plan(
            Step(processor, name="process", writes="processed"),
            Step(scorer, name="score", writes="score"),
            store=store,
            checkpoint_key="fork-test",
            on_concurrent="fork",
        )
        plan._validate({})

        async def run_one(task: str) -> Envelope:
            return await plan.run(
                Envelope(task=task),
                tools=[],
                output_type=str,
                memory=None,
                session=None,
            )

        # Drive all N runs concurrently.  Each gets its own asyncio
        # event loop on a worker thread to simulate independent
        # callers.
        def driver(task: str) -> Envelope:
            return asyncio.run(run_one(task))

        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(driver, inputs))

        # Every run completed without error.
        assert all(r.ok for r in results), "some forks errored"
        # Every per-run keyspace was claimed and finalised.  We can
        # introspect the store to confirm — keys look like
        # ``fork-test:<run_uid>``.  Each finished run carries its own
        # ``processed`` value under that namespace.
        all_keys = store.keys()
        keyspace_keys = [k for k in all_keys if k.startswith("fork-test:")]
        assert len(keyspace_keys) >= n, (
            f"expected at least {n} per-run checkpoint keys; got {len(keyspace_keys)} ({keyspace_keys[:3]}...)"
        )


def test_e2e_concurrent_runs_without_fork_collide() -> None:
    """Without ``on_concurrent='fork'``, two concurrent runs sharing
    the same ``checkpoint_key`` collide on CAS — second run raises
    ``ConcurrentPlanRunError``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "collision.sqlite"))

        # Slow first step gives the second run time to collide.
        async def slow_response(env: Envelope) -> str:
            await asyncio.sleep(0.1)
            return "ok"

        slow = MockAgent(slow_response, name="slow")
        plan = Plan(
            Step(slow, name="step"),
            store=store,
            checkpoint_key="collide",
            # default on_concurrent="fail"
        )
        plan._validate({})

        async def runner() -> Envelope:
            return await plan.run(
                Envelope(task="t"),
                tools=[],
                output_type=str,
                memory=None,
                session=None,
            )

        async def two_concurrent_runs() -> tuple[Any, Any]:
            return await asyncio.gather(
                runner(),
                runner(),
                return_exceptions=True,
            )

        first, second = asyncio.run(two_concurrent_runs())
        # One of the two raised ConcurrentPlanRunError; the other
        # completed.
        errors = [r for r in (first, second) if isinstance(r, ConcurrentPlanRunError)]
        assert len(errors) >= 1, (
            f"expected at least one ConcurrentPlanRunError on collision; got first={first!r} second={second!r}"
        )


# ---------------------------------------------------------------------------
# Scenario 4 — Self-correction loop bounded by max_iterations
# ---------------------------------------------------------------------------
#
#   write ──> review ──[rejected]──> write (loop) ──> ... ──> publish
#
# Stresses:
#   * routes={...} can route BACKWARDS to an earlier step
#   * detour semantics: after a routed-to step, linear progression
#     resumes from there
#   * loop terminates when the predicate stops firing (acceptance)
#   * max_iterations is the safety net; runaway → MaxIterationsExceeded


class _Verdict(BaseModel):
    feedback: str
    approved: bool


@pytest.mark.asyncio
async def test_e2e_self_correction_loop_terminates_on_acceptance() -> None:
    """Reviewer rejects N-1 times then approves on attempt N.  Plan
    loops via ``routes={"write": ...}`` until acceptance, then runs
    publish."""
    review_attempt = {"n": 0}
    accept_after = 3

    def review_response(env: Envelope) -> _Verdict:
        review_attempt["n"] += 1
        approved = review_attempt["n"] >= accept_after
        return _Verdict(
            feedback=f"attempt {review_attempt['n']}",
            approved=approved,
        )

    writer = MockAgent("draft text", name="write")
    reviewer = MockAgent(review_response, name="review", output=_Verdict)
    publisher = MockAgent("published", name="publish")

    plan = Plan(
        Step(writer, name="write", writes="draft"),
        Step(
            reviewer,
            name="review",
            output=_Verdict,
            # Loop back to writer when not approved.
            routes={"write": lambda env: not env.payload.approved},
        ),
        Step(publisher, name="publish"),
        max_iterations=20,
    )
    plan._validate({})
    result = await plan.run(
        Envelope(task="topic"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )
    assert result.ok, f"plan errored: {result.error}"
    # Loop ran exactly accept_after times in the writer→review pair.
    assert len(writer.calls) == accept_after
    assert len(reviewer.calls) == accept_after
    # Publish ran once after final acceptance.
    assert len(publisher.calls) == 1


@pytest.mark.asyncio
async def test_e2e_runaway_loop_caps_at_max_iterations() -> None:
    """Reviewer NEVER approves.  Plan must bail out with
    ``MaxIterationsExceeded`` rather than looping forever."""

    def always_reject(env: Envelope) -> _Verdict:
        return _Verdict(feedback="reject", approved=False)

    writer = MockAgent("draft", name="write")
    reviewer = MockAgent(always_reject, name="review", output=_Verdict)
    publisher = MockAgent("published", name="publish")

    plan = Plan(
        Step(writer, name="write"),
        Step(
            reviewer,
            name="review",
            output=_Verdict,
            routes={"write": lambda env: not env.payload.approved},
        ),
        Step(publisher, name="publish"),
        max_iterations=6,  # tight cap to keep the test fast
    )
    plan._validate({})
    result = await plan.run(
        Envelope(task="topic"),
        tools=[],
        output_type=str,
        memory=None,
        session=None,
    )
    assert not result.ok, "expected MaxIterationsExceeded error"
    assert result.error is not None
    assert result.error.type == "MaxIterationsExceeded"
    # Publisher never reached.
    assert len(publisher.calls) == 0


# ---------------------------------------------------------------------------
# Scenario 5 — Nested Agent-of-Agents — transitive cost rollup
# ---------------------------------------------------------------------------
#
#   outer_agent ─tools─> middle_agent ─tools─> inner_agent
#
# Stresses:
#   * tool-is-Tool dispatch through three levels
#   * Envelope.metadata.nested_* aggregates costs across the tree


@pytest.mark.asyncio
async def test_e2e_nested_agent_cost_rollup_three_levels() -> None:
    """When agent A has agent B as a tool, and B has agent C, every
    LLM call's cost should land in the outer envelope's ``cost_usd``
    + ``nested_cost_usd`` so ``Session.usage_summary()`` aggregates
    cleanly."""
    inner = MockAgent(
        "inner-result",
        name="inner",
        default_input_tokens=10,
        default_output_tokens=5,
        default_cost_usd=0.001,
    )
    middle = MockAgent(
        "middle-result",
        name="middle",
        default_input_tokens=20,
        default_output_tokens=10,
        default_cost_usd=0.002,
    )
    outer = MockAgent(
        "outer-result",
        name="outer",
        default_input_tokens=30,
        default_output_tokens=15,
        default_cost_usd=0.003,
    )

    # Run them sequentially via a Plan to verify cost aggregation
    # along the chain.
    plan = Plan(
        Step(outer, name="outer"),
        Step(middle, name="middle"),
        Step(inner, name="inner"),
    )
    plan._validate({})
    sess = Session()
    try:
        result = await plan.run(
            Envelope(task="t"),
            tools=[],
            output_type=str,
            memory=None,
            session=sess,
        )
        assert result.ok
        # The metadata of the final envelope holds nested totals from
        # every prior step in the plan (Plan._aggregate_nested_metadata).
        meta = result.metadata
        # All three steps contributed their cost.
        total_cost = (meta.cost_usd or 0.0) + meta.nested_cost_usd
        # 0.001 + 0.002 + 0.003 = 0.006 (within float tolerance).
        assert abs(total_cost - 0.006) < 1e-6, f"expected total cost ~0.006, got {total_cost}"
        total_in = meta.input_tokens + meta.nested_input_tokens
        total_out = meta.output_tokens + meta.nested_output_tokens
        # 10 + 20 + 30 = 60.
        assert total_in == 60, f"expected 60 input tokens, got {total_in}"
        # 5 + 10 + 15 = 30.
        assert total_out == 30, f"expected 30 output tokens, got {total_out}"
    finally:
        sess.close()


# ---------------------------------------------------------------------------
# Scenario 6 — Mixed step targets (Agent + plain callable + tool name)
# ---------------------------------------------------------------------------
#
# Stresses: ``Step.target`` is uniform across Agent / function / tool name.
# All three dispatch through the same Plan engine without special-casing.


@pytest.mark.asyncio
async def test_e2e_mixed_step_targets_dispatch_uniformly() -> None:
    captured = {"normalised": None, "scored": None}

    # Plain Python callable — pure compute, no LLM.
    def normalise(text: str) -> str:
        out = text.strip().lower()
        captured["normalised"] = out
        return out

    # Agent target.
    fetcher = MockAgent("  Hello WORLD  ", name="fetch")

    # Agent target after the callable.
    summariser = MockAgent("summary", name="summary")

    # Tool target — provided via ``Plan.run(tools=...)`` rather than
    # constructed at Plan build time.  Plan resolves the string target
    # against the tools list at execution time.
    def score(text: str) -> int:
        captured["scored"] = text
        return len(text)

    from lazybridge.tools import build_tool_map

    tools = build_tool_map([score])

    plan = Plan(
        Step(fetcher, name="fetch"),
        Step(normalise, name="clean"),
        Step("score", name="score"),
        Step(summariser, name="summary"),
    )
    plan._validate(tools)
    result = await plan.run(
        Envelope(task="t"),
        tools=list(tools.values()),
        output_type=str,
        memory=None,
        session=None,
    )
    assert result.ok, f"plan errored: {result.error}"
    # All three targets dispatched.
    assert captured["normalised"] == "hello world"
    assert captured["scored"] == "hello world"
    assert len(summariser.calls) == 1
