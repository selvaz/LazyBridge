"""Executable proof for docs/guides/project-patterns.md.

Every claim the cookbook marks [verified] is asserted here. If LazyBridge
changes such that a pattern stops being true, this suite fails and the
cookbook gets corrected instead of quietly rotting.

Run:  pytest test_project_patterns.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from lazybridge import Agent, Envelope, Plan, Step, Store, Tool
from lazybridge.engines.plan import PlanCompileError, PlanPaused
from lazybridge.sentinels import from_agent, from_memory, from_parallel_all, from_step
from lazybridge.testing import MockAgent


@pytest.fixture
def tmp_store(tmp_path: Path) -> Store:
    return Store(db=str(tmp_path / "s.sqlite"))


# ===========================================================================
# §1.1 — a plain Python function is a legal Step target
# ===========================================================================
def test_pure_function_pipeline_runs_without_an_llm() -> None:
    def fetch(task: str) -> str:
        return f"fetched({task})"

    def transform(task: str) -> str:
        return f"transformed[{task}]"

    env = Agent(engine=Plan(Step(fetch), Step(transform)), name="p")("seed")

    assert env.ok
    assert env.text() == "transformed[fetched(seed)]"


def test_step_by_tool_name_resolves_via_tool_wrap() -> None:
    def enrich(task: str) -> str:
        """Enrich the incoming task."""
        return f"enriched({task})"

    env = Agent(
        engine=Plan(Step("enrich")),
        tools=[Tool.wrap(enrich, name="enrich")],
        name="p",
    )("seed")

    assert env.text() == "enriched(seed)"


# ===========================================================================
# §1.2 — nesting, and its atomicity cost
# ===========================================================================
def test_nested_subplan_is_a_legal_step_target() -> None:
    sub = Agent(
        engine=Plan(
            Step(lambda t: f"A({t})", name="a"),
            Step(lambda t: f"B({t})", name="b"),
        ),
        name="sub",
    )
    env = Agent(
        engine=Plan(Step(sub, name="sub"), Step(lambda t: f"tail[{t}]", name="tail")),
        name="outer",
    )("x")

    assert env.text() == "tail[B(A(x))]"


def test_two_unnamed_lambdas_collide_at_compile_time() -> None:
    """Gotcha: a Step's default name is target.__name__ — every lambda is '<lambda>'."""
    with pytest.raises(PlanCompileError, match="duplicate step name"):
        Agent(engine=Plan(Step(lambda t: "a"), Step(lambda t: "b")), name="p")


def test_nested_subplan_reruns_from_the_top_on_parent_resume(tmp_store: Store) -> None:
    """A sub-plan is ATOMIC to the parent checkpoint — §1.2's trade-off."""
    calls: list[str] = []
    fail = {"on": True}

    def inner_1(task: str) -> str:
        calls.append("inner_1")
        return "i1"

    def inner_2(task: str) -> str:
        calls.append("inner_2")
        if fail["on"]:
            raise RuntimeError("inner boom")
        return "i2"

    def build(resume: bool) -> Agent:
        sub = Agent(engine=Plan(Step(inner_1), Step(inner_2)), name="sub")
        return Agent(
            engine=Plan(
                Step(sub, name="sub"),
                Step(lambda t: "done", name="tail"),
                store=tmp_store,
                checkpoint_key="nest",
                resume=resume,
            ),
            name="outer",
        )

    assert not build(False)("go").ok
    fail["on"] = False
    calls.clear()
    assert build(True)("go").ok

    # inner_1 had already succeeded, yet it runs again.
    assert calls[:2] == ["inner_1", "inner_2"]


# ===========================================================================
# §1.3 — routing needs a predicate for EVERY branch
# ===========================================================================
def _routed(routes: dict, trace: list[str]) -> Agent:
    def classify(task: str) -> str:
        trace.append("classify")
        return "urgent" if "fire" in task else "normal"

    def urgent(task: str) -> str:
        trace.append("urgent")
        return "URGENT"

    def normal(task: str) -> str:
        trace.append("normal")
        return "NORMAL"

    def archive(task: str) -> str:
        trace.append("archive")
        return f"archived:{task}"

    return Agent(
        engine=Plan(
            Step(classify, name="classify", routes=routes, after_branches="archive"),
            Step(urgent, name="urgent"),
            Step(normal, name="normal"),
            Step(archive, name="archive"),
        ),
        name="r",
    )


def test_incomplete_routes_fall_through_and_run_both_branches() -> None:
    trace: list[str] = []
    only_urgent = {"urgent": lambda e: e.text() == "urgent"}
    _routed(only_urgent, trace)("routine check")

    # The documented trap: the 'normal' input also executes the urgent handler.
    assert trace == ["classify", "urgent", "normal", "archive"]


def test_a_predicate_per_branch_selects_exactly_one() -> None:
    both = {
        "urgent": lambda e: e.text() == "urgent",
        "normal": lambda e: e.text() == "normal",
    }

    t1: list[str] = []
    _routed(both, t1)("fire in the datacenter")
    assert t1 == ["classify", "urgent", "archive"]

    t2: list[str] = []
    _routed(both, t2)("routine check")
    assert t2 == ["classify", "normal", "archive"]


# ===========================================================================
# §1.4 — checkpoint keys and resume
# ===========================================================================
def test_a_permanent_key_plus_resume_makes_later_runs_noops(tmp_store: Store) -> None:
    runs: list[str] = []

    def work(task: str) -> str:
        runs.append(task)
        return "result"

    def nightly() -> Agent:
        return Agent(
            engine=Plan(
                Step(work, writes="out"),
                store=tmp_store,
                checkpoint_key="nightly",  # <-- the bug: permanent key
                resume=True,
            ),
            name="nightly",
        )

    nightly()("day1")
    nightly()("day2")
    nightly()("day3")

    assert len(runs) == 1, "a fixed key + resume=True silently one-shots the job"


def test_resume_reruns_only_the_failed_step(tmp_store: Store) -> None:
    calls: list[str] = []
    fail = {"on": True}

    def one(task: str) -> str:
        calls.append("one")
        return "one-done"

    def two(task: str) -> str:
        calls.append("two")
        if fail["on"]:
            raise RuntimeError("boom")
        return "two-done"

    def build(resume: bool) -> Agent:
        return Agent(
            engine=Plan(
                Step(one, writes="one"),
                Step(two, writes="two"),
                store=tmp_store,
                checkpoint_key="run-1",
                resume=resume,
            ),
            name="p",
        )

    assert not build(False)("go").ok
    fail["on"] = False
    assert build(True)("go").ok
    assert calls == ["one", "two", "two"]


def test_writes_lands_in_the_durable_store(tmp_store: Store) -> None:
    Agent(
        engine=Plan(
            Step(lambda t: "VALUE", name="s", writes="saved"),
            store=tmp_store,
            checkpoint_key="w",
        ),
        name="p",
    )("go")

    assert tmp_store.read("saved") == "VALUE"


# ===========================================================================
# §1.5 — a failed plan RETURNS; it does not raise
# ===========================================================================
def test_a_failing_plan_returns_not_ok_instead_of_raising() -> None:
    def explode(task: str) -> str:
        raise RuntimeError("kaboom")

    env = Agent(engine=Plan(Step(explode)), name="boom")("go")

    assert env.ok is False
    assert env.error is not None


# ===========================================================================
# §1.6 — fan-out and parallel bands
# ===========================================================================
def test_run_many_returns_input_order_even_when_completion_order_differs() -> None:
    """Forces completion order to be the REVERSE of input order.

    Without forcing completion order this test would pass even if
    `run_many` returned results in completion order, since same-cost tasks
    finish in submission order anyway. Ordering is forced with an event
    chain rather than staggered sleeps, so it cannot flake under scheduler
    load: "third" always appends first and signals "second", which always
    appends next and signals "first" — the sequence is enforced by the
    events themselves, not by timing margins.
    """
    import threading

    completion: list[str] = []
    barrier = threading.Barrier(3)  # all three workers must be running first
    third_done = threading.Event()
    second_done = threading.Event()

    # Bounded, not indefinite: a concurrency regression (run_many starting
    # fewer than 3 workers, or one never reaching its signal) must fail this
    # test, not hang the CI job until the whole run times out.
    _TIMEOUT = 10

    def score(task: str) -> str:
        barrier.wait(timeout=_TIMEOUT)
        if task == "third":
            completion.append(task)
            third_done.set()
        elif task == "second":
            assert third_done.wait(timeout=_TIMEOUT), "third never signaled — concurrency regression"
            completion.append(task)
            second_done.set()
        else:  # "first"
            assert second_done.wait(timeout=_TIMEOUT), "second never signaled — concurrency regression"
            completion.append(task)
        return f"scored:{task}"

    tasks = ["first", "second", "third"]
    envs = Plan(Step(score, name="score")).run_many(tasks, concurrency=3)

    assert completion == ["third", "second", "first"], (
        "the test must actually invert completion order to prove anything"
    )
    assert [e.text() for e in envs] == [
        "scored:first",
        "scored:second",
        "scored:third",
    ], "results must follow INPUT order, not completion order"


def test_run_many_isolates_a_failing_task_to_its_own_slot() -> None:
    """One bad task must not raise, drop a slot, or fail its siblings."""

    def score(task: str) -> str:
        if task == "BAD":
            raise RuntimeError("no such ticker")
        return f"scored:{task}"

    envs = Plan(Step(score, name="score")).run_many(["AAPL", "BAD", "TLT"], concurrency=3)

    assert len(envs) == 3
    assert envs[1].ok is False and envs[1].error is not None
    assert envs[0].ok and envs[2].ok
    assert envs[0].text() == "scored:AAPL" and envs[2].text() == "scored:TLT"


def test_from_parallel_all_aggregates_a_band() -> None:
    env = Agent(
        engine=Plan(
            Step(lambda t: "PRICES", name="prices", parallel=True),
            Step(lambda t: "NEWS", name="news", parallel=True),
            Step(lambda t: f"combined<<{t}>>", name="combine", task=from_parallel_all("prices")),
        ),
        name="agg",
    )("go")

    assert "PRICES" in env.text() and "NEWS" in env.text()


def test_a_failed_parallel_band_applies_no_sibling_writes(tmp_store: Store) -> None:
    def good(task: str) -> str:
        return "GOOD"

    def bad(task: str) -> str:
        raise RuntimeError("branch down")

    env = Agent(
        engine=Plan(
            Step(good, name="good", parallel=True, writes="good"),
            Step(bad, name="bad", parallel=True, writes="bad"),
            store=tmp_store,
            checkpoint_key="band",
        ),
        name="band",
    )("go")

    assert not env.ok
    assert tmp_store.read("good") is None, "band must be atomic"
    assert tmp_store.read("bad") is None


# ===========================================================================
# §1.7 — PlanPaused
# ===========================================================================
def test_planpaused_checkpoints_the_same_step_and_resumes_there(tmp_store: Store) -> None:
    gate = {"ready": False}
    calls: list[str] = []

    def before(task: str) -> str:
        calls.append("before")
        return "prepared"

    def wait_for_it(task: str) -> str:
        calls.append("wait")
        if not gate["ready"]:
            raise PlanPaused("waiting for settlement")
        return "settled"

    def after(task: str) -> str:
        calls.append("after")
        return "done"

    def build(resume: bool) -> Agent:
        return Agent(
            engine=Plan(
                Step(before, name="before"),
                Step(wait_for_it, name="wait"),
                Step(after, name="after"),
                store=tmp_store,
                checkpoint_key="pause-run",
                resume=resume,
            ),
            name="paused",
        )

    first = build(False)("go")
    assert "PlanPaused" in str(first.error)
    assert calls == ["before", "wait"]

    gate["ready"] = True
    calls.clear()
    assert build(True)("go").ok
    assert calls == ["wait", "after"], "resume re-invokes the SAME step"


# ===========================================================================
# §2.1 — where the typed payload survives, and where it collapses to str
# ===========================================================================
BIG = {"series": list(range(500))}


def test_a_plain_function_target_receives_only_a_string() -> None:
    seen: list[str] = []

    def observe(task) -> str:
        seen.append(type(task).__name__)
        return "ok"

    Agent(engine=Plan(Step(lambda t: BIG), Step(observe)), name="p")("go")

    assert seen == ["str"], "raw callables get env.task, never the payload"


def test_an_agent_target_receives_the_typed_payload() -> None:
    received: list = []

    class SpyEngine:
        model = "spy"

        async def run(self, env, **kw):
            received.append(env.payload)
            return Envelope(task=env.task, payload="spied")

        async def stream(self, env, **kw):  # pragma: no cover
            yield ""

    spy = Agent(engine=SpyEngine(), name="spy")
    Agent(engine=Plan(Step(lambda t: BIG, name="produce"), Step(spy, name="spy")), name="p")("go")

    assert isinstance(received[0], dict)
    assert len(received[0]["series"]) == 500


def test_handle_not_payload_keeps_the_inter_step_task_tiny() -> None:
    depot: dict[str, object] = {}
    seen_handle: list[str] = []

    def load(task: str) -> str:
        depot["run42"] = BIG
        return "run42"

    def consume(key: str) -> str:
        seen_handle.append(key)
        return f"rows={len(depot[key.strip()]['series'])}"  # type: ignore[index]

    env = Agent(engine=Plan(Step(load), Step(consume)), name="p")("start")

    assert env.text() == "rows=500"
    assert len(seen_handle[0]) < 20


# ===========================================================================
# §2.3 — writes= is NOT a sentinel-readable channel
# ===========================================================================
def test_writes_bucket_is_not_readable_by_a_sentinel() -> None:
    with pytest.raises(PlanCompileError, match="not in the tool map"):
        Agent(
            engine=Plan(
                Step(lambda t: "V", name="h", writes="saved"),
                Step(lambda t: t, name="r", task=from_memory("saved")),
            ),
            name="p",
        )


def test_from_step_skips_intermediate_steps() -> None:
    env = Agent(
        engine=Plan(
            Step(lambda t: "HEAD", name="head"),
            Step(lambda t: "MIDDLE", name="middle"),
            Step(lambda t: f"saw={t}", name="tail", task=from_step("head")),
        ),
        name="p",
    )("go")

    assert env.text() == "saw=HEAD"


def test_from_agent_fails_open_two_different_ways(tmp_store: Store) -> None:
    """§2.3's `from_agent` warning, both shapes: never-written IS empty, but
    a stale prior success is NOT — presence alone can't tell them apart."""
    calls = {"n": 0}

    def body(task: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "first-run-output"
        raise RuntimeError("boom")

    source = Agent(engine=Plan(Step(body, name="body")), name="source", store=tmp_store)

    def consume(task: str) -> str:
        return f"consumed={task!r}"

    reader = Agent(
        engine=Plan(Step(consume, name="consume", task=from_agent("source"))),
        tools=[source.as_tool("source")],
        store=tmp_store,
        name="reader",
    )

    # Shape 1: key never written yet — resolves to an empty envelope.
    assert reader("go").text() == "consumed=''"

    # First run succeeds and writes the store key.
    assert source("go").ok
    assert reader("go").text() == "consumed='first-run-output'"

    # Shape 2: current run's source agent fails — the store key is left
    # untouched, so from_agent silently returns the STALE first-run value,
    # not empty. A presence check on this value would pass anyway.
    assert not source("go").ok, "second run must actually fail, or this proves nothing"
    assert reader("go").text() == "consumed='first-run-output'"


# ===========================================================================
# §3.1 — the artifact registry is a convention over Store
# ===========================================================================
def test_registry_discovery_provenance_and_cas(tmp_store: Store) -> None:
    tmp_store.write("artifact:anomaly:daily:2026-08-17:abc", {"rows": 12}, agent_id="daily")
    tmp_store.write("artifact:anomaly:daily:2026-08-16:def", {"rows": 9}, agent_id="daily")
    tmp_store.write("unrelated:thing", {"x": 1}, agent_id="other")

    found = [k for k in tmp_store if k.startswith("artifact:anomaly:daily:")]
    assert len(found) == 2

    entry = tmp_store.read_entry("artifact:anomaly:daily:2026-08-17:abc")
    assert entry.agent_id == "daily", "agent_id provenance survives the round-trip"

    tmp_store.write("head:anomaly", "rev1")
    assert tmp_store.compare_and_swap("head:anomaly", "rev1", "rev2") is True
    assert tmp_store.compare_and_swap("head:anomaly", "rev1", "rev3") is False
    assert tmp_store.read("head:anomaly") == "rev2"


def test_reserve_before_acting_admits_exactly_one_sender(tmp_store: Store) -> None:
    """§3.2 — claiming the key BEFORE the side effect is what bounds it.

    Two workers racing on the same report key: exactly one may proceed.
    """
    key = "sent:2026-08-17:report-1"

    first_claim = tmp_store.compare_and_swap(key, None, "in-flight")
    second_claim = tmp_store.compare_and_swap(key, None, "in-flight")

    assert first_claim is True
    assert second_claim is False, "a second worker must not also send"

    tmp_store.write(key, "confirmed")
    assert tmp_store.read(key) == "confirmed"


def test_a_separate_PROCESS_sees_the_same_artifacts(tmp_path: Path) -> None:
    """§3.1 claims cross-process visibility, so read it from a real subprocess."""
    import json
    import subprocess
    import sys

    db = str(tmp_path / "shared.sqlite")
    Store(db=db).write("artifact:x", {"v": 1}, agent_id="producer")

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys;from lazybridge import Store;print(json.dumps(Store(db=sys.argv[1]).read('artifact:x')))",
            db,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip()) == {"v": 1}


# ===========================================================================
# §4.1 / §4.2 — reliability and test doubles
# ===========================================================================
def test_fallback_fires_on_a_returned_error_envelope() -> None:
    class ErrEngine:
        model = "err"

        async def run(self, env, **kw):
            return Envelope.error_envelope(RuntimeError("primary down"))

        async def stream(self, env, **kw):  # pragma: no cover
            yield ""

    primary = Agent(
        engine=ErrEngine(),
        name="primary",
        fallback=MockAgent(responses=["backup answer"], name="backup"),
    )

    assert "backup answer" in primary("hello").text()


def test_fallback_does_not_catch_an_ENGINE_level_raise() -> None:
    """The documented boundary is the *engine*, not a tool: a custom engine
    that raises propagates past fallback. (A Plan STEP that raises is
    converted to an error envelope and DOES trigger it — see the test below.)
    """

    class RaisingEngine:
        model = "raise"

        async def run(self, env, **kw):
            raise RuntimeError("primary exploded")

        async def stream(self, env, **kw):  # pragma: no cover
            raise RuntimeError("primary exploded")
            yield ""

    agent = Agent(
        engine=RaisingEngine(),
        name="raiser",
        fallback=MockAgent(responses=["backup"], name="backup2"),
    )

    with pytest.raises(RuntimeError, match="primary exploded"):
        agent("hello")


def test_fallback_DOES_cover_an_exception_raised_inside_a_plan_step() -> None:
    """A step that raises becomes an error envelope, so fallback fires."""

    def exploding_step(task: str) -> str:
        raise RuntimeError("step exploded")

    backup = MockAgent(responses=["BACKUP RAN"], name="backup")
    agent = Agent(
        engine=Plan(Step(exploding_step, name="boom")),
        name="planned",
        fallback=backup,
    )

    env = agent("go")

    assert env.ok
    assert env.text() == "BACKUP RAN"
    backup.assert_call_count(1)


# ===========================================================================
# §4.3 — an approval gate must not execute on rejection
# ===========================================================================
def _gate(verdict: str, trace: list[str]) -> Agent:
    def prepare(task: str) -> str:
        trace.append("prepare")
        return "orders-ready"

    def approver(task: str) -> str:
        trace.append("approve")
        return verdict

    def execute_orders(task: str) -> str:
        trace.append("execute")
        return "orders executed"

    def halt(task: str) -> str:
        trace.append("halt")
        return "halted"

    def finish(task: str) -> str:
        trace.append("finish")
        return f"final: {task}"

    return Agent(
        engine=Plan(
            Step(prepare, name="prepare"),
            Step(
                approver,
                name="approve",
                routes={
                    "execute": lambda e: e.text().strip().lower() == "approve",
                    "halt": lambda e: e.text().strip().lower() != "approve",
                },
                after_branches="finish",
            ),
            Step(execute_orders, name="execute"),
            Step(halt, name="halt"),
            Step(finish, name="finish"),
        ),
        name="gate",
    )


def test_approval_gate_does_not_execute_on_rejection() -> None:
    trace: list[str] = []
    _gate("reject: too risky", trace)("go")

    assert "execute" not in trace, "a rejection must never execute the orders"
    assert trace == ["prepare", "approve", "halt", "finish"]


def test_approval_gate_executes_once_on_approval() -> None:
    trace: list[str] = []
    _gate("approve", trace)("go")

    assert trace == ["prepare", "approve", "execute", "finish"]


def test_approval_gate_rejects_ambiguous_near_miss_text() -> None:
    """`startswith("approve")` would fire on this — exact match must not."""
    trace: list[str] = []
    _gate("approve? no", trace)("go")

    assert "execute" not in trace, "a near-miss that merely starts with 'approve' must not execute"
    assert trace == ["prepare", "approve", "halt", "finish"]


def test_naive_gate_without_a_reject_predicate_executes_anyway() -> None:
    """The documented trap — kept as a test so the doc's warning stays true."""
    trace: list[str] = []

    def approver(task: str) -> str:
        return "reject: too risky"

    def execute_orders(task: str) -> str:
        trace.append("execute")
        return "orders executed"

    Agent(
        engine=Plan(
            Step(approver, name="approve", routes={"execute": lambda e: e.text().startswith("approve")}),
            Step(execute_orders, name="execute"),
        ),
        name="naive",
    )("go")

    assert trace == ["execute"], "documents why a predicate per branch is mandatory"


def test_human_engine_output_lives_on_agent_and_step_not_the_engine() -> None:
    """§4.3's structured-decision fix: `HumanEngine(output=...)` is not a
    thing — the schema goes on the Agent wrapping it and on the Step, and
    `routes_by=`'s Literal values must BE step names (no predicate layer
    remaps them)."""
    from pydantic import BaseModel

    from lazybridge.ext.hil import HumanEngine

    class Decision(BaseModel):
        decision: Literal["execute", "halt"]

    class _FakeUI:
        def __init__(self, reply: str) -> None:
            self.reply = reply

        async def prompt(self, task: str, *, tools: list, output_type: type) -> str:
            return self.reply

    trace: list[str] = []

    def execute_orders(task: str) -> str:
        trace.append("execute")
        return "orders executed"

    def halt(task: str) -> str:
        trace.append("halt")
        return "halted"

    def build(reply: str) -> Agent:
        engine = HumanEngine(default='{"decision":"halt"}')
        engine._ui = _FakeUI(reply)
        approval = Agent(engine=engine, output=Decision, name="approve")
        return Agent(
            engine=Plan(
                Step(approval, name="approve", output=Decision, routes_by="decision", after_branches="finish"),
                Step(execute_orders, name="execute"),
                Step(halt, name="halt"),
                Step(lambda t: "done", name="finish"),
            ),
            name="gate",
        )

    trace.clear()
    assert build('{"decision":"halt"}')("go").ok
    assert trace == ["halt"]

    trace.clear()
    assert build('{"decision":"execute"}')("go").ok
    assert trace == ["execute"]


def test_routes_by_literal_that_is_not_a_step_name_fails_at_compile_time() -> None:
    """A `Literal["approve", "reject"]` naming (rather than real step names)
    is the mistake the doc calls out — caught before the plan ever runs."""
    from pydantic import BaseModel

    class Decision(BaseModel):
        decision: Literal["approve", "reject"]

    with pytest.raises(PlanCompileError, match="not a known step name"):
        Agent(
            engine=Plan(
                Step(lambda t: t, name="approve", output=Decision, routes_by="decision"),
                Step(lambda t: t, name="execute"),
                Step(lambda t: t, name="halt"),
            ),
            name="gate",
        )


# ===========================================================================
# §4.4 — Session is not inherited by a Step-target agent
# ===========================================================================
def _sub_with_emitting_engine(name: str, session=None) -> Agent:
    """A sub-agent whose engine actually emits AGENT_START.

    Engines emit the agent span, not `Agent` — so a silent stub engine would
    make this test fail for the wrong reason regardless of session wiring.
    `Plan` is the cheapest engine that genuinely emits.
    """
    return Agent(
        engine=Plan(Step(lambda t: "sub-out", name="inner")),
        name=name,
        session=session,
    )


def _agents_with_own_span(sess) -> set[str]:
    """Agents that emitted their OWN agent_start span into this session.

    Plan tags its per-step tool_call/tool_result with the *parent's* name, so
    filtering on agent_start is what distinguishes "this agent reported for
    itself" from "the parent mentioned it as a step".
    """
    return {
        row["payload"]["agent_name"]
        for row in sess.events.query()
        if row.get("event_type") == "agent_start"
        and isinstance(row.get("payload"), dict)
        and row["payload"].get("agent_name")
    }


def test_step_target_agent_events_do_not_reach_the_root_session() -> None:
    """Not just `.session is None` — its events are genuinely absent."""
    from lazybridge import Session

    sess = Session()
    sub = _sub_with_emitting_engine("sub_uninherited")  # no session=
    Agent(engine=Plan(Step(sub, name="sub_uninherited")), name="root", session=sess)("go")
    sess.flush()

    assert sub.session is None
    names = _agents_with_own_span(sess)
    assert "root" in names, "the root agent must appear, or this test proves nothing"
    assert "sub_uninherited" not in names, "sub-agent reported its own span — the doc's warning is now stale"


def test_explicit_session_puts_the_sub_agent_back_in_the_log() -> None:
    """The documented fix must actually close the hole the test above shows."""
    from lazybridge import Session

    sess = Session()
    sub = _sub_with_emitting_engine("sub_explicit", session=sess)  # explicit
    Agent(engine=Plan(Step(sub, name="sub_explicit")), name="root2", session=sess)("go")
    sess.flush()

    assert sub.session is sess
    assert "sub_explicit" in _agents_with_own_span(sess)


def test_mockagent_drives_a_plan_and_records_the_wiring() -> None:
    research = MockAgent(responses=["research-key"], name="research")
    write = MockAgent(responses=["done"], name="write")

    result = Agent(
        engine=Plan(Step(research, name="research"), Step(write, name="write")),
        name="pipe",
    )("run-42")

    assert result.ok
    research.assert_call_count(1)
    write.assert_called_with(contains="research-key")
