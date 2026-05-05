"""Tool-is-Tool completeness — every factory result usable as a tool.

The framework's "Tool-is-Tool" contract says: anything constructed
through ``Agent`` (or the ext factories) plugs into ``tools=[...]`` of
ANOTHER agent with identical semantics — the LLM sees a tool that
takes a string task and returns a stringifiable result.

Pre-fix gap: ``Agent.from_parallel`` returned ``_ParallelAgent``
whose ``run()`` resolved to ``list[Envelope]`` (not ``Envelope``).
The Tool wrapper claimed ``returns_envelope=True`` but the actual
return was a list, so the LLM's tool-result content path silently
fell through ``str(list)`` — opaque junk in the model's context.

Fix: ``_ParallelAgent.as_tool()`` folds the N branches into ONE
``Envelope`` via labelled-text join (same shape as ``Plan``'s
``from_parallel_all`` aggregator) with transitive cost rollup and
first-error short-circuit.

This module pins the contract for every documented factory.
"""

from __future__ import annotations

import asyncio

from lazybridge import Agent, Step
from lazybridge.envelope import Envelope, ErrorInfo
from lazybridge.testing import MockAgent
from lazybridge.tools import Tool, wrap_tool

# ---------------------------------------------------------------------------
# Every factory's result wraps cleanly via wrap_tool()
# ---------------------------------------------------------------------------


def test_from_chain_wraps_as_tool():
    a = MockAgent("A", name="a_agent")
    b = MockAgent("B", name="b_agent")
    chained = Agent.from_chain(a, b)
    t = wrap_tool(chained)
    assert isinstance(t, Tool)


def test_from_plan_wraps_as_tool():
    plan_agent = Agent.from_plan(Step(target=lambda task: "x", name="step1"))
    t = wrap_tool(plan_agent)
    assert isinstance(t, Tool)


def test_from_parallel_wraps_as_tool():
    fan = Agent.from_parallel(MockAgent("X"), MockAgent("Y"), name="fan")
    t = wrap_tool(fan)
    assert isinstance(t, Tool)
    assert t.name == "fan"


def test_supervisor_agent_wraps_as_tool():
    from lazybridge.ext.hil import supervisor_agent

    sup = supervisor_agent(tools=[], agents=[], name="sup")
    t = wrap_tool(sup)
    assert isinstance(t, Tool)


def test_human_agent_wraps_as_tool():
    from lazybridge.ext.hil import human_agent

    hu = human_agent(timeout=1.0, name="approve")
    t = wrap_tool(hu)
    assert isinstance(t, Tool)


def test_orchestrator_agent_wraps_as_tool():
    from lazybridge.ext.planners import orchestrator_agent

    sub = MockAgent("sub-out", name="sub")
    orch = orchestrator_agent(agents=[sub])
    t = wrap_tool(orch)
    assert isinstance(t, Tool)
    assert t.name == "planner"


def test_blackboard_orchestrator_agent_wraps_as_tool():
    from lazybridge.ext.planners import blackboard_orchestrator_agent

    sub = MockAgent("sub-out", name="sub")
    bb = blackboard_orchestrator_agent(agents=[sub])
    t = wrap_tool(bb)
    assert isinstance(t, Tool)
    assert t.name == "blackboard_planner"


# ---------------------------------------------------------------------------
# Parallel-as-tool: invoking the wrapped tool returns ONE Envelope
# ---------------------------------------------------------------------------


def test_parallel_as_tool_returns_single_envelope_not_list():
    """The fix: ``as_tool()`` must fold ``list[Envelope]`` into ONE
    ``Envelope`` so the Tool contract holds.  Pre-fix, ``run()`` was a
    list and the LLM tool-result block stringified it as junk."""
    fan = Agent.from_parallel(MockAgent("X-out"), MockAgent("Y-out"), name="fan")
    t = wrap_tool(fan)
    result = asyncio.run(t.run(task="hi"))
    assert isinstance(result, Envelope)


def test_parallel_as_tool_payload_is_labelled_text_join():
    """Same shape as ``Plan._aggregate_parallel_band`` — every branch
    appears as ``[name]\\n<output>`` separated by blank lines."""
    fan = Agent.from_parallel(
        MockAgent("first-output", name="first"),
        MockAgent("second-output", name="second"),
        name="fan",
    )
    t = wrap_tool(fan)
    env = asyncio.run(t.run(task="hi"))
    text = env.text()
    assert "[first]" in text
    assert "first-output" in text
    assert "[second]" in text
    assert "second-output" in text
    # Sanity: the join is blank-line separated.
    assert "\n\n" in text


def test_parallel_as_tool_aggregates_nested_metadata():
    """Cost rolls up transitively — every branch's tokens / cost are
    summed into the wrapper's ``nested_*`` so an outer agent reading
    ``.metadata.nested_input_tokens`` sees the full spend."""
    fan = Agent.from_parallel(MockAgent("A"), MockAgent("B"), name="fan")
    t = wrap_tool(fan)
    env = asyncio.run(t.run(task="hi"))
    # MockAgent reports input_tokens=10 / output_tokens=20 per call.
    assert env.metadata.nested_input_tokens == 20
    assert env.metadata.nested_output_tokens == 40


def test_parallel_as_tool_first_error_propagates():
    """If any branch errors, the wrapper's ``.error`` is set so
    downstream short-circuit detection works."""

    class _BrokenAgent:
        _is_lazy_agent = True
        name = "broken"
        description = None
        session = None

        async def run(self, task):
            return Envelope(
                task=str(task),
                error=ErrorInfo(type="Boom", message="branch failed"),
            )

    fan = Agent.from_parallel(MockAgent("ok"), _BrokenAgent(), name="fan")  # type: ignore[arg-type]
    t = wrap_tool(fan)
    env = asyncio.run(t.run(task="hi"))
    assert env.error is not None
    assert env.error.type == "Boom"


def test_parallel_as_tool_custom_name_and_description():
    fan = Agent.from_parallel(MockAgent("A"), MockAgent("B"))
    custom = fan.as_tool(name="my_fan", description="run my fan-out")
    assert custom.name == "my_fan"
    # The description shows up on the Tool's definition.
    assert "run my fan-out" in (custom.description or "")


# ---------------------------------------------------------------------------
# End-to-end: parallel-as-tool plugged into another Agent's tools=
# ---------------------------------------------------------------------------


def test_parallel_as_tool_pluggable_into_outer_tools_list():
    """The full Tool-is-Tool contract: an outer Agent constructed
    with ``tools=[fan]`` must recognise the fan-out runner as a
    legitimate tool, register it in its tool map, and not crash at
    construction.  (We don't fire a real LLM here — we just verify
    construction-time tool registration succeeds.)"""
    fan = Agent.from_parallel(MockAgent("A"), MockAgent("B"), name="fanout")
    sub = MockAgent("just-a-tool", name="aux")

    # Outer agent built around an LLMEngine (not invoked) — registers
    # both ``fan`` and ``sub`` in its tool map.  If wrap_tool fails on
    # ``fan``, this construction would crash.
    outer = Agent("claude-opus-4-7", tools=[fan, sub])
    tool_names = {t.name for t in outer._tool_map.values()}
    assert "fanout" in tool_names
    assert "aux" in tool_names


def test_parallel_as_tool_definition_advertises_string_task_input():
    """The Tool's definition must advertise a ``task: str`` input —
    same shape as every other agent-as-tool.  This is what lets an
    LLM emit a tool call with a single ``task`` argument."""
    fan = Agent.from_parallel(MockAgent("A"), MockAgent("B"), name="fan")
    t = wrap_tool(fan)
    schema = t.definition().parameters
    props = schema.get("properties", {})
    assert "task" in props, f"expected 'task' parameter on the parallel tool, got {list(props)}"


# ---------------------------------------------------------------------------
# Boundary: the underlying _ParallelAgent.run() still returns list[Envelope]
# (so direct callers — Agent.from_parallel(a, b)("task") — keep working)
# ---------------------------------------------------------------------------


def test_parallel_direct_call_still_returns_list_envelope():
    """The ``as_tool()`` wrapper folds — but the underlying runner's
    ``__call__`` still returns ``list[Envelope]`` (the documented
    asymmetry of scripted fan-out).  Pre-existing callers must not
    silently break."""
    fan = Agent.from_parallel(MockAgent("A"), MockAgent("B"))
    out = fan("hi")
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(e, Envelope) for e in out)
