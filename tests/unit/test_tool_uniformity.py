"""Tool-is-Tool uniformity: nested Agent/function composition, and the
"parallelism is a capability, not a config" contract.

What we verify:

* LLMEngine executes every tool call via ``asyncio.gather`` regardless
  of ``tool_choice`` — the old "parallel" mode is no longer a user-facing
  knob.  Legacy callers passing ``tool_choice='parallel'`` get a
  ``DeprecationWarning`` and the value is collapsed to ``'auto'``.
* ``Tool.run_sync`` handles both sync and async ``func``, so an Agent
  wrapped by ``Agent.as_tool`` can be invoked through a REPL-style
  engine (SupervisorEngine) without hitting the raw coroutine.
* A three-level nest —
  ``outer(Agent) → middle(Agent) → leaf(function)`` — composes and runs
  through the SupervisorEngine path without special-casing.  Session
  propagates through all three levels; the graph records the chain.
* ``Agent.parallel`` stays: deterministic fan-out returning
  ``list[Envelope]``, documented as sugar for ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest

from lazybridge import (
    Agent,
    LLMEngine,
    Session,
    SupervisorEngine,
    Tool,
)


# ---------------------------------------------------------------------------
# LLMEngine — parallel tool execution is now a capability, not a config
# ---------------------------------------------------------------------------


def test_llmengine_tool_choice_literal_no_longer_accepts_parallel_type():
    """The type annotation narrows the accepted literals to auto / any.
    Runtime still tolerates 'parallel' for backward compat but warns.
    """
    from typing import get_args, get_type_hints

    hints = get_type_hints(LLMEngine.__init__)
    tc_literal = hints.get("tool_choice")
    assert tc_literal is not None
    accepted = set(get_args(tc_literal))
    assert accepted == {"auto", "any"}


def test_llmengine_accepts_parallel_legacy_with_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        eng = LLMEngine("claude-opus-4-7", tool_choice="parallel")  # type: ignore[arg-type]
    assert eng.tool_choice == "auto"
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    assert any("parallel" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# Tool.run_sync handles async funcs — needed for Agent.as_tool through REPLs
# ---------------------------------------------------------------------------


def test_tool_run_sync_on_async_func_returns_value_not_coroutine():
    async def slow_echo(q: str) -> str:
        await asyncio.sleep(0)
        return f"async-echo:{q}"

    tool = Tool(slow_echo, name="slow_echo")
    out = tool.run_sync(q="hi")
    assert out == "async-echo:hi"


def test_tool_run_sync_on_sync_func_unchanged():
    def echo(q: str) -> str:
        return f"sync-echo:{q}"

    tool = Tool(echo, name="echo")
    assert tool.run_sync(q="hi") == "sync-echo:hi"


def test_tool_run_sync_on_agent_as_tool_returns_envelope_with_text():
    """Agent.as_tool now returns the inner agent's full Envelope.

    The REPL caller must (a) never see a raw coroutine, (b) be able to
    reach the string via ``str(...)`` / ``.text()`` unchanged.
    """
    from lazybridge.envelope import Envelope

    class _FakeEngine:
        async def run(self, env, *, tools, output_type, memory, session):
            return Envelope(task=env.task, payload=f"fake:{env.task}")

        async def stream(self, *a, **kw):  # pragma: no cover
            if False:
                yield ""

    inner = Agent(engine=_FakeEngine(), name="inner")
    inner_tool = inner.as_tool("inner", "inner agent")

    out = inner_tool.run_sync(task="hello")
    assert isinstance(out, Envelope)
    assert out.text() == "fake:hello"
    assert str(out) == "fake:hello"   # Envelope.__str__ safety net
    assert inner_tool.returns_envelope is True


# ---------------------------------------------------------------------------
# Tool-is-Tool uniformity: function → agent → agent-of-agent
# ---------------------------------------------------------------------------


class _EchoEngine:
    """Minimal fake engine that returns ``f'{label}:<task>'`` — lets us build
    deterministic Agent chains without touching real provider APIs.
    """

    def __init__(self, label: str) -> None:
        self._label = label

    async def run(self, env, *, tools, output_type, memory, session):
        from lazybridge.envelope import Envelope
        # If there's exactly one tool, call it with the task — simulates an
        # LLM agent that decided to delegate.  Otherwise just echo.
        if len(tools) == 1:
            result = await tools[0].run(**{_first_param(tools[0]): env.task})
            return Envelope(task=env.task, payload=f"{self._label}:{result}")
        return Envelope(task=env.task, payload=f"{self._label}:{env.task}")

    async def stream(self, *a, **kw):  # pragma: no cover
        if False:
            yield ""


def _first_param(tool: Tool) -> str:
    params = tool.definition().parameters.get("properties", {})
    return next(iter(params), "task")


def test_nested_agent_of_agent_of_function_runs_uniformly():
    def leaf(query: str) -> str:
        """Leaf tool."""
        return f"leaf({query})"

    # Build: outer(Agent) → middle(Agent) → leaf(function)
    middle = Agent(engine=_EchoEngine("middle"), tools=[leaf], name="middle")
    outer = Agent(engine=_EchoEngine("outer"), tools=[middle], name="outer")

    env = outer("x")
    # outer's engine delegates to its single tool (middle.as_tool via wrap_tool);
    # middle's engine delegates to its single tool (leaf);
    # leaf returns 'leaf(x)'.
    assert env.text() == "outer:middle:leaf(x)"


def test_nested_agents_share_outer_session_by_default():
    """Session propagation (fix #2) must work at every depth of the tree."""
    sess = Session()

    def leaf(query: str) -> str:
        return query

    middle = Agent(engine=_EchoEngine("m"), tools=[leaf], name="middle")
    outer = Agent(engine=_EchoEngine("o"), tools=[middle], name="outer",
                  session=sess)

    # Middle inherited outer's session; leaf is a plain callable so it
    # doesn't have one.
    assert middle.session is sess

    # Both agents are visible in the session's graph, with an as_tool edge.
    names = {n.name for n in sess.graph.nodes()}
    assert {"outer", "middle"}.issubset(names)
    edges = [(e.from_id, e.to_id, e.label) for e in sess.graph.edges()]
    assert ("outer", "middle", "as_tool") in edges


# ---------------------------------------------------------------------------
# Same tools=[...] surface across engines
# ---------------------------------------------------------------------------


def _scripted(lines: list[str]):
    it = iter(lines)
    return lambda _prompt: next(it)


def test_same_tools_list_works_on_llm_and_supervisor_engines():
    """The exact same tools list is accepted by LLMEngine and SupervisorEngine
    Agents without API divergence.  Only observable behaviour differs —
    the LLM decides when to invoke tools, the supervisor lets a human.
    """
    def calc(x: str) -> str:
        """Return x doubled."""
        return f"calc({x})"

    tools = [calc]

    llm_agent = Agent(engine=_EchoEngine("llm"), tools=tools, name="llm")
    sup_agent = Agent(
        engine=SupervisorEngine(tools=tools, input_fn=_scripted(["calc(42)", "continue"])),
        name="sup",
    )

    # LLM path: _EchoEngine auto-delegates to the single tool.
    assert llm_agent("42").text() == "llm:calc(42)"

    # Supervisor path: the human-scripted REPL calls calc(42) then continues.
    assert sup_agent("42").text() == "calc(42)"


# ---------------------------------------------------------------------------
# Agent.parallel preserved — deterministic fan-out, list[Envelope]
# ---------------------------------------------------------------------------


def test_agent_parallel_is_deterministic_fanout():
    a = Agent(engine=_EchoEngine("a"), name="a")
    b = Agent(engine=_EchoEngine("b"), name="b")
    c = Agent(engine=_EchoEngine("c"), name="c")

    results = Agent.parallel(a, b, c)("hello")
    assert len(results) == 3
    texts = [r.text() for r in results]
    assert texts == ["a:hello", "b:hello", "c:hello"]
