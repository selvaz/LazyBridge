"""End-to-end stress scenarios — complex pipelines run for real.

Each scenario builds a non-trivial Plan and runs it through the actual
engine + Store + Session, with deterministic ``MockAgent`` responses
standing in for LLM calls.  Asserts framework invariants that unit
tests can't catch (cost roll-up across nested agents, atomicity under
errors, isolation under concurrent runs, resume correctness after a
simulated crash).

Style note
==========
These tests deliberately avoid every ``async``/``await``/
``asyncio.run``/``ThreadPoolExecutor`` idiom — that's exactly the
boilerplate LazyBridge was built to hide.  Plans are dispatched via
the sync ``Agent.from_engine(plan)("task")`` façade and via
``plan.run_many(...)`` for fan-out; routing predicates use the
``when`` DSL.  Read these scenarios as if they were sample apps,
not framework internals.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from lazybridge import Agent, Plan, Step, Store, when
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
#   * routes={...} with the ``when`` DSL evaluated correctly on real Envelope
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
            routes={"apology": when.field("items").empty()},
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


def test_e2e_research_happy_path_runs_full_linear_then_terminates() -> None:
    """When search returns items, the predicate doesn't fire → linear all
    the way through publish.  Apology IS reached via linear fall-through
    after publish (it's the next declared step); this is the documented
    Pattern B caveat."""
    plan, agents = _build_research_plan(["paper-1", "paper-2", "paper-3"])
    with Session() as sess:
        result = Agent.from_engine(plan, session=sess)("AI trends April 2026")
        assert result.ok, f"plan errored: {result.error}"
        assert len(agents["search"].calls) == 1
        assert len(agents["rank"].calls) == 1
        assert len(agents["analyse"].calls) == 1
        assert len(agents["publish"].calls) == 1
        # Apology IS reached via linear fall-through after publish.
        assert len(agents["apology"].calls) == 1


def test_e2e_research_early_out_routes_directly_to_apology() -> None:
    """When search returns no items, the predicate fires → jump directly
    to apology.  Rank/analyse/publish never run."""
    plan, agents = _build_research_plan(items_returned=[])
    with Session() as sess:
        result = Agent.from_engine(plan, session=sess)("obscure query")
        assert result.ok
        assert len(agents["search"].calls) == 1
        assert len(agents["rank"].calls) == 0, "rank should be skipped on early-out"
        assert len(agents["analyse"].calls) == 0
        assert len(agents["publish"].calls) == 0
        assert len(agents["apology"].calls) == 1


def test_e2e_cost_rollup_aggregates_across_all_steps() -> None:
    """Session events must capture every step's token / cost usage."""
    plan, _agents = _build_research_plan(["item"])
    with Session() as sess:
        Agent.from_engine(plan, session=sess)("t")
        # The Plan engine emits TOOL_CALL / TOOL_RESULT around each step.
        tool_calls = sess.events.query(event_type=EventType.TOOL_CALL)
        tool_results = sess.events.query(event_type=EventType.TOOL_RESULT)
        # 5 steps declared; happy path runs all 5.
        assert len(tool_calls) == 5
        assert len(tool_results) == 5
        step_names_called = {tc["payload"]["step"] for tc in tool_calls}
        assert step_names_called == {"search", "rank", "analyse", "publish", "apology"}


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


def test_e2e_parallel_band_atomicity_under_branch_error() -> None:
    """If one branch in a parallel band raises, the framework must:
    (1) emit the band's failure as the result,
    (2) NOT apply ``writes`` from any branch (atomic)."""
    load = MockAgent("loaded", name="load")

    def branch_a_response(env: Envelope) -> _ParallelOut:
        return _ParallelOut(finding="A-result")

    def branch_b_response(env: Envelope) -> _ParallelOut:
        # Simulate a transient failure
        raise RuntimeError("simulated branch B crash")

    def branch_c_response(env: Envelope) -> _ParallelOut:
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

        result = Agent.from_engine(plan)("t")
        assert not result.ok, "expected error envelope from failed band"

        # Atomicity: no band writes were applied.
        assert store.read("loaded") is not None
        assert store.read("band_a") is None, "branch A's write leaked despite band failure"
        assert store.read("band_b") is None
        assert store.read("band_c") is None
        # synth and finalise never ran.
        assert len(synth.calls) == 0
        assert len(finalise.calls) == 0


def test_e2e_parallel_band_aggregates_via_from_parallel_all() -> None:
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
    Agent.from_engine(plan)("t")
    # All three branch outputs ended up in synth's context.
    ctx = captured_context[0]
    assert "alpha-result" in ctx
    assert "beta-result" in ctx
    assert "gamma-result" in ctx


def test_e2e_resume_picks_up_at_failed_step_after_simulated_crash() -> None:
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
        result1 = Agent.from_engine(plan1)("batch")
        assert not result1.ok, "expected first run to fail at transform"
        # extract's write survived; transform's didn't.
        assert store.read("raw") == "raw-data"
        assert store.read("clean") is None
        # extract called once; transform attempted once and failed
        # (MockAgent.calls records only successful runs).
        assert len(extract.calls) == 1
        assert attempts["transform"] == 1, "transform should have been attempted once"
        assert len(load.calls) == 0

        # Run 2: resume.
        plan2 = make_plan(store)
        plan2._validate({})
        result2 = Agent.from_engine(plan2)("batch")
        assert result2.ok, f"expected resume to succeed; got {result2.error}"
        assert len(extract.calls) == 1, "extract should have been skipped on resume"
        assert attempts["transform"] == 2, "transform should have been re-attempted once"
        assert len(transform.calls) == 1, "transform succeeded exactly once (the resume)"
        assert len(load.calls) == 1


# ---------------------------------------------------------------------------
# Scenario 3 — N concurrent fork runs (no asyncio / ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def test_e2e_concurrent_fork_runs_have_isolated_keyspaces() -> None:
    """Run N=8 copies of the same pipeline concurrently with
    ``on_concurrent='fork'`` via ``Plan.run_many``.  Each run gets its
    own ``f"{checkpoint_key}:{run_uid}"`` namespace.  No
    ``ThreadPoolExecutor``, no ``asyncio.run`` in the test code."""
    n = 8
    inputs = [f"task-{i}" for i in range(n)]

    def processor_response(env: Envelope) -> str:
        return f"processed-{env.task}"

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "fork.sqlite"))
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

        results = plan.run_many(inputs, concurrency=n)
        assert all(r.ok for r in results), "some forks errored"
        keyspace_keys = [k for k in store if k.startswith("fork-test:")]
        assert len(keyspace_keys) >= n, f"expected at least {n} per-run checkpoint keys; got {len(keyspace_keys)}"


def test_e2e_concurrent_runs_without_fork_collide() -> None:
    """Without ``on_concurrent='fork'``, two concurrent runs sharing
    the same ``checkpoint_key`` collide on CAS — ``run_many`` returns
    error envelopes (containing ``ConcurrentPlanRunError``) for the
    losing run(s)."""
    import asyncio  # only inside this test for a slow MockAgent response

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Store(db=str(Path(tmpdir) / "collision.sqlite"))

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

        results = plan.run_many(["a", "b"])
        # At least one of the two failed with ConcurrentPlanRunError.
        errored = [r for r in results if not r.ok]
        assert errored, "expected at least one collision error"
        assert any(r.error is not None and "ConcurrentPlanRunError" in r.error.type for r in errored)


# ---------------------------------------------------------------------------
# Scenario 4 — Self-correction loop bounded by max_iterations
# ---------------------------------------------------------------------------


class _Verdict(BaseModel):
    feedback: str
    approved: bool


def test_e2e_self_correction_loop_terminates_on_acceptance() -> None:
    """Reviewer rejects N-1 times then approves on attempt N.  Plan
    loops via ``routes={"write": when.field("approved").is_(False)}``
    until acceptance, then runs publish."""
    review_attempt = {"n": 0}
    accept_after = 3

    def review_response(env: Envelope) -> _Verdict:
        review_attempt["n"] += 1
        approved = review_attempt["n"] >= accept_after
        return _Verdict(feedback=f"attempt {review_attempt['n']}", approved=approved)

    writer = MockAgent("draft text", name="write")
    reviewer = MockAgent(review_response, name="review", output=_Verdict)
    publisher = MockAgent("published", name="publish")

    plan = Plan(
        Step(writer, name="write", writes="draft"),
        Step(
            reviewer,
            name="review",
            output=_Verdict,
            routes={"write": when.field("approved").is_(False)},
        ),
        Step(publisher, name="publish"),
        max_iterations=20,
    )
    plan._validate({})
    result = Agent.from_engine(plan)("topic")
    assert result.ok, f"plan errored: {result.error}"
    assert len(writer.calls) == accept_after
    assert len(reviewer.calls) == accept_after
    assert len(publisher.calls) == 1


def test_e2e_runaway_loop_caps_at_max_iterations() -> None:
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
            routes={"write": when.field("approved").is_(False)},
        ),
        Step(publisher, name="publish"),
        max_iterations=6,
    )
    plan._validate({})
    result = Agent.from_engine(plan)("topic")
    assert not result.ok, "expected MaxIterationsExceeded error"
    assert result.error is not None
    assert result.error.type == "MaxIterationsExceeded"
    assert len(publisher.calls) == 0


# ---------------------------------------------------------------------------
# Scenario 5 — Nested cost rollup
# ---------------------------------------------------------------------------


def test_e2e_nested_agent_cost_rollup_three_levels() -> None:
    """When three steps each report distinct cost/token figures, the
    final envelope's ``cost_usd + nested_cost_usd`` totals the entire
    pipeline spend exactly."""
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

    plan = Plan(
        Step(outer, name="outer"),
        Step(middle, name="middle"),
        Step(inner, name="inner"),
    )
    plan._validate({})
    with Session() as sess:
        result = Agent.from_engine(plan, session=sess)("t")
        assert result.ok
        meta = result.metadata
        total_cost = (meta.cost_usd or 0.0) + meta.nested_cost_usd
        assert abs(total_cost - 0.006) < 1e-6, f"expected total cost ~0.006, got {total_cost}"
        total_in = meta.input_tokens + meta.nested_input_tokens
        total_out = meta.output_tokens + meta.nested_output_tokens
        assert total_in == 60, f"expected 60 input tokens, got {total_in}"
        assert total_out == 30, f"expected 30 output tokens, got {total_out}"


# ---------------------------------------------------------------------------
# Scenario 6 — Mixed step targets (Agent + plain callable + tool name)
# ---------------------------------------------------------------------------


def test_e2e_mixed_step_targets_dispatch_uniformly() -> None:
    captured: dict[str, str | None] = {"normalised": None, "scored": None}

    def normalise(text: str) -> str:
        out = text.strip().lower()
        captured["normalised"] = out
        return out

    fetcher = MockAgent("  Hello WORLD  ", name="fetch")
    summariser = MockAgent("summary", name="summary")

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
    # The Agent façade hides the tools= / output_type= / etc. plumbing.
    result = Agent.from_engine(plan, tools=list(tools.values()))("t")
    assert result.ok, f"plan errored: {result.error}"
    assert captured["normalised"] == "hello world"
    assert captured["scored"] == "hello world"
    assert len(summariser.calls) == 1
