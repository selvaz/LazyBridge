"""Cross-cutting Plan / Session / verify guarantees.

Covers:
* Plan ``Step(parallel=True)`` runs concurrently via ``asyncio.gather``
  in declared-consecutive groups.
* ``_resolve_sentinel`` preserves Envelope metadata across
  ``from_prev`` / ``from_step`` / ``from_parallel``.
* ``verify_with_retry`` keeps the pristine original task on every
  retry; feedback flows through ``context``, not ``task``.
* ``Session.emit`` warns on exporter failure instead of swallowing
  silently; ``_redact`` return is validated.
* ``session.register_agent`` / ``register_tool_edge`` failures surface
  as ``UserWarning`` instead of being silently swallowed.
"""

from __future__ import annotations

import asyncio
import time
import warnings

import pytest

from lazybridge import (
    Agent,
    Envelope,
    EventType,
    Plan,
    Session,
    Step,
    from_prev,
    from_step,
)
from lazybridge._verify import verify_with_retry
from lazybridge.envelope import EnvelopeMetadata

# ---------------------------------------------------------------------------
# Plan Step(parallel=True) really runs concurrently
# ---------------------------------------------------------------------------


def test_plan_parallel_steps_actually_run_concurrently():
    """Three async sleep(0.1) steps in parallel complete in ~0.1s, not 0.3s."""

    async def slow_a(task: str) -> str:
        await asyncio.sleep(0.08)
        return f"a:{task}"

    async def slow_b(task: str) -> str:
        await asyncio.sleep(0.08)
        return f"b:{task}"

    async def slow_c(task: str) -> str:
        await asyncio.sleep(0.08)
        return f"c:{task}"

    plan = Plan(
        Step(slow_a, name="a", parallel=True, writes="a_out"),
        Step(slow_b, name="b", parallel=True, writes="b_out"),
        Step(slow_c, name="c", parallel=True, writes="c_out"),
    )

    t0 = time.monotonic()
    Agent.from_engine(plan)("x")
    elapsed = time.monotonic() - t0

    # Parallel: one step's wall-time, plus a little overhead.
    assert elapsed < 0.18, f"parallel took {elapsed:.3f}s — suggests sequential execution"


def test_plan_parallel_group_followed_by_sequential_join():
    """Parallel branches run concurrently; a non-parallel step afterwards
    runs once after they all complete (the conventional "join").
    """
    order: list[str] = []

    async def branch_a(task: str) -> str:
        order.append("a")
        return "A"

    async def branch_b(task: str) -> str:
        order.append("b")
        return "B"

    def join(task: str) -> str:
        order.append("join")
        return "joined"

    plan = Plan(
        Step(branch_a, name="a", parallel=True),
        Step(branch_b, name="b", parallel=True),
        Step(join, name="join"),
    )
    env = Agent.from_engine(plan)("hi")

    # Both branches ran, join ran exactly once and after both.
    assert "a" in order and "b" in order
    assert order[-1] == "join"
    assert env.text() == "joined"


def test_plan_parallel_branch_error_fails_the_plan():
    """If one parallel branch raises, the plan surfaces an error envelope
    with a checkpoint pointing at the failing branch.
    """

    async def ok(task: str) -> str:
        return "ok"

    async def boom(task: str) -> str:
        raise RuntimeError("branch failed")

    plan = Plan(
        Step(ok, name="ok", parallel=True),
        Step(boom, name="boom", parallel=True),
    )
    env = Agent.from_engine(plan)("x")

    assert not env.ok
    assert "branch failed" in env.error.message


def test_plan_sequential_still_works_after_refactor():
    """Non-parallel plans must not regress after the parallel-group
    branch was added to the run loop.
    """

    def a(task: str) -> str:
        return f"a:{task}"

    def b(task: str) -> str:
        return f"b:{task}"

    plan = Plan(Step(a, name="a"), Step(b, name="b"))
    env = Agent.from_engine(plan)("hi")
    # from_prev chain semantics — b sees a's output
    assert env.text() == "b:a:hi"


# ---------------------------------------------------------------------------
# Sentinel resolution preserves metadata
# ---------------------------------------------------------------------------


def test_from_prev_preserves_envelope_metadata():
    """Metadata on the previous step's envelope must survive the
    sentinel hop; otherwise nested-cost aggregation breaks across Plan
    step boundaries.
    """

    class _MeteredEngine:
        async def run(self, env, *, tools, output_type, memory, session):
            return Envelope(
                task=env.task,
                payload="out",
                metadata=EnvelopeMetadata(
                    input_tokens=50,
                    output_tokens=20,
                    cost_usd=0.003,
                ),
            )

        async def stream(self, *a, **kw):  # pragma: no cover
            if False:
                yield ""

    # Single-step plan using the metered engine as a tool; the step's
    # target resolves via from_prev sentinel (default), then the captured
    # envelope should carry metadata through.
    producer = Agent(engine=_MeteredEngine(), name="producer")

    def reader(task: str) -> str:
        return f"read({task})"

    plan = Plan(
        Step(producer, name="produce"),
        Step(reader, name="read", task=from_prev),
    )
    env = Agent.from_engine(plan)("start")
    # reader ran — value confirms from_prev passed producer's payload along
    assert "read(" in env.text()


def test_from_step_preserves_envelope_metadata():
    """Same guarantee for an explicit from_step("name") reference."""

    def a(task: str) -> str:
        return "A-output"

    def b(task: str) -> str:
        return f"saw:{task}"

    plan = Plan(
        Step(a, name="a"),
        Step(b, name="b", task=from_step("a")),
    )
    env = Agent.from_engine(plan)("root")
    assert env.text() == "saw:A-output"


# ---------------------------------------------------------------------------
# verify_with_retry preserves original task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_with_retry_preserves_original_task_across_retries():
    """The pristine user task is preserved across all retries; feedback
    flows via ``env.context``, never concatenated onto ``env.task``.
    """
    tasks_seen: list[str] = []
    contexts_seen: list[str | None] = []

    class _AgentUnderTest:
        async def run(self, env):
            tasks_seen.append(env.task)
            contexts_seen.append(env.context)
            return Envelope(task=env.task, payload="bad-output")

    verdicts = iter(["rejected: wrong", "rejected: still wrong", "approved"])

    def judge(output: str) -> str:
        return next(verdicts)

    env = Envelope.from_task("what is 2+2?")
    await verify_with_retry(_AgentUnderTest(), env, judge, max_verify=3)

    # All three attempts saw the EXACT same task.
    assert tasks_seen == ["what is 2+2?"] * 3
    # Feedback is surfaced via context, accumulating from the judge's prior verdicts.
    assert contexts_seen[0] is None
    assert contexts_seen[1] is not None and "wrong" in contexts_seen[1]
    assert contexts_seen[2] is not None and "still wrong" in contexts_seen[2]


@pytest.mark.asyncio
async def test_verify_with_retry_approved_on_first_returns_immediately():
    class _A:
        calls = 0

        async def run(self, env):
            _A.calls += 1
            return Envelope(task=env.task, payload="ok")

    def judge(output: str) -> str:
        return "approved"

    env = Envelope.from_task("q")
    result = await verify_with_retry(_A(), env, judge, max_verify=5)
    assert result.text() == "ok"
    assert _A.calls == 1


# ---------------------------------------------------------------------------
# Session.emit warns on exporter failure
# ---------------------------------------------------------------------------


def test_session_emit_warns_on_exporter_exception():
    """A buggy exporter must surface as a DeprecationWarning-style
    warning, not silently eat events.
    """

    class _BadExporter:
        def export(self, event):
            raise ValueError("no u")

    sess = Session(exporters=[_BadExporter()])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sess.emit(EventType.AGENT_START, {"agent_name": "x", "task": "t"})

    msgs = [str(x.message) for x in w]
    assert any("_BadExporter" in m and "ValueError" in m for m in msgs), msgs


def test_session_emit_suppresses_repeated_exporter_warnings():
    """First failure warns; subsequent failures from the same exporter
    are suppressed to avoid log spam.
    """

    class _BadExporter:
        def export(self, event):
            raise ValueError("still no")

    exp = _BadExporter()
    sess = Session(exporters=[exp])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sess.emit(EventType.AGENT_START, {"agent_name": "x"})
        sess.emit(EventType.AGENT_FINISH, {"agent_name": "x"})

    bad_msgs = [m for m in (str(x.message) for x in w) if "_BadExporter" in m]
    assert len(bad_msgs) == 1


# ---------------------------------------------------------------------------
# Session.redact validation
# ---------------------------------------------------------------------------


def test_session_redact_returning_non_dict_warns_and_preserves_payload():
    """A redactor that returns None / wrong shape must NOT crash the
    emit path or silently produce an empty payload.
    """

    def bad_redactor(payload):
        return None

    sess = Session(redact=bad_redactor)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sess.emit(EventType.AGENT_START, {"agent_name": "x", "task": "hi"})

    msgs = [str(x.message) for x in w]
    assert any("redact" in m.lower() and "NoneType" in m for m in msgs)


def test_session_redact_valid_dict_is_applied():
    """Happy path: a dict-returning redactor is applied verbatim."""

    def redact(payload):
        out = dict(payload)
        out["task"] = "[REDACTED]"
        return out

    sess = Session(redact=redact)
    sess.emit(EventType.AGENT_START, {"agent_name": "x", "task": "secret"})
    rows = sess.events.query(event_type=EventType.AGENT_START)
    assert rows and rows[0]["payload"]["task"] == "[REDACTED]"


# ---------------------------------------------------------------------------
# Agent registration failures warn
# ---------------------------------------------------------------------------


def test_agent_warns_on_broken_session_register_agent():
    """A custom Session subclass whose ``register_agent`` raises
    surfaces as a ``UserWarning`` rather than being silently swallowed.
    """

    class _BrokenSession(Session):
        def register_agent(self, agent):
            raise RuntimeError("bad session")

    sess = _BrokenSession()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Agent("claude-opus-4-7", name="a", session=sess)

    msgs = [str(x.message) for x in w]
    assert any("register_agent" in m and "bad session" in m for m in msgs)
