"""Tests for features carried forward from the pre-v1 branch.

Covers:
  * ``Session.graph`` — auto-registration of agents as the graph is built.
  * ``Agent.as_tool(verify=, max_verify=)`` — tool-level judge/retry (Option B).
  * ``Session(console=True)`` / ``Agent(verbose=True)`` — stdout event tracing
    via :class:`~lazybridge.ConsoleExporter`.
  * ``SupervisorEngine`` — human-in-the-loop REPL with tool calls, agent retry,
    and store access (scripted ``input_fn`` for test determinism).
  * ``Plan`` checkpoint / resume via a backing ``Store``.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from lazybridge import (
    Agent,
    ConsoleExporter,
    EdgeType,
    Envelope,
    EventType,
    GraphSchema,
    Plan,
    Session,
    Step,
    Store,
    SupervisorEngine,
    Tool,
)


# ---------------------------------------------------------------------------
# Session.graph — auto-registration
# ---------------------------------------------------------------------------


def test_session_exposes_graph_by_default():
    sess = Session()
    assert isinstance(sess.graph, GraphSchema)
    assert sess.graph.session_id == sess.session_id


def test_agent_auto_registers_into_session_graph():
    sess = Session()
    Agent("claude-opus-4-7", name="researcher", session=sess)
    Agent("gpt-4o", name="writer", session=sess)

    nodes = sess.graph.nodes()
    names = {n.name for n in nodes}
    assert names == {"researcher", "writer"}

    # Provider inferred from the model string and exposed by LLMEngine.
    by_name = {n.name: n for n in nodes}
    assert by_name["researcher"].provider == "anthropic"
    assert by_name["researcher"].model == "claude-opus-4-7"
    assert by_name["writer"].provider == "openai"


def test_session_register_tool_edge_links_agents():
    sess = Session()
    a = Agent("claude-opus-4-7", name="a", session=sess)
    b = Agent("claude-opus-4-7", name="b", session=sess)
    sess.register_tool_edge(a, b, label="handoff")

    edges = sess.graph.edges()
    assert len(edges) == 1
    assert edges[0].from_id == "a"
    assert edges[0].to_id == "b"
    assert edges[0].kind == EdgeType.TOOL
    assert edges[0].label == "handoff"


# ---------------------------------------------------------------------------
# Session(console=True) / Agent(verbose=True) — stdout event tracing
# ---------------------------------------------------------------------------


def test_session_console_installs_console_exporter():
    sess = Session(console=True)
    assert any(isinstance(e, ConsoleExporter) for e in sess._exporters)


def test_console_exporter_writes_event_to_stream():
    buf = io.StringIO()
    exp = ConsoleExporter(stream=buf)
    exp.export({
        "event_type": "tool_call",
        "agent_name": "researcher",
        "session_id": "ignored",
        "task": "hello",
    })
    out = buf.getvalue()
    assert "[researcher]" in out
    assert "tool_call" in out
    assert "task=hello" in out
    assert "session_id" not in out  # noisy keys filtered


def test_agent_verbose_creates_private_session_with_console():
    # Passing verbose= without a session= should spin up a private Session
    # that has a ConsoleExporter attached.
    ag = Agent("claude-opus-4-7", name="demo", verbose=True)
    assert ag.session is not None
    assert any(isinstance(e, ConsoleExporter) for e in ag.session._exporters)


# ---------------------------------------------------------------------------
# Agent.as_tool(verify=, max_verify=) — tool-level judge (Option B)
# ---------------------------------------------------------------------------


class _FakeAgent:
    """Minimal Agent-like object for tool-level verify tests."""

    _is_lazy_agent = True

    def __init__(self, outputs: list[str], *, name: str = "fake") -> None:
        self._outputs = outputs
        self._i = 0
        self.name = name
        self.description = None

    async def run(self, task: Any) -> Envelope:  # type: ignore[override]
        out = self._outputs[min(self._i, len(self._outputs) - 1)]
        self._i += 1
        if isinstance(task, Envelope):
            return Envelope(task=task.task, payload=out)
        return Envelope(task=str(task), payload=out)


async def _run_tool(tool: Tool, task: str) -> str:
    result = await tool.run(task=task)
    # Post-Envelope-preservation change: Agent.as_tool returns an
    # Envelope.  Tests written against the flat-string contract stay
    # green because ``str(envelope)`` routes through ``__str__ →
    # text()`` and yields the same string.
    return str(result)


@pytest.mark.asyncio
async def test_as_tool_without_verify_returns_agent_output():
    agent = _FakeAgent(["first"])
    tool = Agent.as_tool(agent, "fake", "desc")  # type: ignore[arg-type]
    assert await _run_tool(tool, "hello") == "first"


@pytest.mark.asyncio
async def test_as_tool_with_judge_retries_until_approved():
    # Agent under test produces 'bad', 'bad', 'good'
    agent = _FakeAgent(["bad", "bad", "good"])

    # Judge: plain callable — approves only the literal 'good'
    def judge(output: str) -> str:
        return "approved" if output == "good" else "rejected: try again"

    tool = Agent.as_tool(agent, "fake", "desc", verify=judge, max_verify=5)  # type: ignore[arg-type]

    out = await _run_tool(tool, "task")
    assert out == "good"
    # Judge called four times total: three agent invocations, the last of
    # which gets approved. Agent ran 3 times.
    assert agent._i == 3


@pytest.mark.asyncio
async def test_as_tool_with_judge_gives_up_after_max_verify():
    agent = _FakeAgent(["bad", "bad", "bad"])

    def judge(output: str) -> str:
        return "rejected: always bad"

    tool = Agent.as_tool(agent, "fake", "desc", verify=judge, max_verify=2)  # type: ignore[arg-type]

    out = await _run_tool(tool, "task")
    # After max_verify attempts the last attempt is returned as-is.
    assert out == "bad"
    assert agent._i == 2


# ---------------------------------------------------------------------------
# SupervisorEngine — interactive HIL (scripted input_fn)
# ---------------------------------------------------------------------------


def _scripted_input(lines: list[str]):
    it = iter(lines)

    def _fn(_prompt: str) -> str:
        return next(it)

    return _fn


def test_supervisor_continue_returns_last_output():
    sess = Session()
    sup = Agent(
        engine=SupervisorEngine(input_fn=_scripted_input(["continue"])),
        name="sup",
        session=sess,
    )
    env = sup("hello task")
    assert env.text() == "hello task"


def test_supervisor_tool_call_uses_tool_output():
    def echo(query: str) -> str:
        """Return the query prefixed with 'echo:'."""
        return f"echo:{query}"

    echo_tool = Tool(echo)

    sup = Agent(
        engine=SupervisorEngine(
            tools=[echo_tool],
            input_fn=_scripted_input(["echo(hello)", "continue"]),
        ),
        name="sup",
    )
    env = sup("x")
    assert env.text() == "echo:hello"


def test_supervisor_retry_reruns_registered_agent_with_feedback():
    # Record what the fake agent was prompted with on each call.
    calls: list[str] = []

    class _Ag:
        name = "researcher"
        def __call__(self, task: str) -> str:
            calls.append(task)
            return f"<r:{len(calls)}>"

    sup = Agent(
        engine=SupervisorEngine(
            agents=[_Ag()],
            input_fn=_scripted_input(["retry researcher: include 2026", "continue"]),
        ),
        name="sup",
    )
    env = sup("initial task")
    assert env.text() == "<r:1>"
    # Retry appends the feedback to the original task.
    assert "initial task" in calls[0]
    assert "include 2026" in calls[0]


def test_supervisor_store_access_reads_shared_store(capsys):
    store = Store()
    store.write("doc", "important text")

    sup = Agent(
        engine=SupervisorEngine(
            store=store,
            input_fn=_scripted_input(["store doc", "continue"]),
        ),
        name="sup",
    )
    sup("x")
    out = capsys.readouterr().out
    assert "Store[doc]: important text" in out


# ---------------------------------------------------------------------------
# Plan checkpoint / resume
# ---------------------------------------------------------------------------


def test_plan_writes_checkpoint_to_store_after_each_step():
    store = Store()

    def s1(task: str) -> str:
        return "one"

    def s2(task: str) -> str:
        return "two"

    plan = Plan(
        Step(s1, writes="a"),
        Step(s2, writes="b"),
        store=store,
        checkpoint_key="ck",
    )
    Agent(engine=plan)("go")

    saved = store.read("ck")
    assert saved["status"] == "done"
    assert saved["kv"] == {"a": "one", "b": "two"}
    assert saved["completed_steps"] == ["s1", "s2"]
    assert saved["next_step"] is None


def test_plan_resume_after_failure_retries_only_failed_step():
    store = Store()

    run_counts = {"s1": 0, "s2": 0, "s3": 0}
    def s1(task: str) -> str:
        run_counts["s1"] += 1
        return "a"
    def s2(task: str) -> str:
        run_counts["s2"] += 1
        return "b"
    def s3(task: str) -> str:
        run_counts["s3"] += 1
        if run_counts["s3"] < 2:
            raise RuntimeError("transient")
        return "c"

    def build():
        return Plan(
            Step(s1, writes="r1"),
            Step(s2, writes="r2"),
            Step(s3, writes="r3"),
            store=store,
            checkpoint_key="ck",
            resume=True,
        )

    # First run — s3 fails.
    env1 = Agent(engine=build())("go")
    assert env1.error is not None
    saved = store.read("ck")
    assert saved["status"] == "failed"
    assert saved["next_step"] == "s3"

    # Second run — s1/s2 are skipped, s3 retries and succeeds.
    env2 = Agent(engine=build())("go")
    assert env2.error is None
    assert run_counts == {"s1": 1, "s2": 1, "s3": 2}
    saved2 = store.read("ck")
    assert saved2["status"] == "done"


def test_plan_resume_after_done_returns_cached_kv_without_rerun():
    store = Store()
    calls = {"s1": 0}

    def s1(task: str) -> str:
        calls["s1"] += 1
        return "only"

    def build():
        return Plan(
            Step(s1, writes="r1"),
            store=store,
            checkpoint_key="ck",
            resume=True,
        )

    Agent(engine=build())("go")
    Agent(engine=build())("go")  # should NOT re-run

    assert calls["s1"] == 1
