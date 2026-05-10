"""Regression test for B9 — blackboard planner must reset its closure
state between Agent invocations.

Background: ``make_blackboard_planner`` builds a closure-held ``state``
dict shared by ``set_plan``, ``get_plan``, ``mark_done``.  Before this
fix, calling the same planner twice in one session leaked the prior
plan into the second run.  The fix subclasses ``Agent`` and resets the
closure state inside ``run()``.
"""

from __future__ import annotations

import asyncio

from lazybridge import Agent, Envelope
from lazybridge.ext.planners.blackboard import make_blackboard_planner


class _CapturingMockEngine:
    """Minimal Engine that ignores tools and returns an empty Envelope.

    The blackboard test only needs to verify that state resets across
    runs; we don't need the planner LLM to actually call the closure
    tools — we exercise them directly via the planner's tool map.
    Accepts arbitrary ``**kwargs`` so it's compatible with the real
    Engine protocol regardless of which keyword arguments the Agent
    forwards (output_type, memory, session, store, plan_state, ...).
    """

    model = "mock-model"

    async def run(self, env: Envelope, **kwargs) -> Envelope:
        return Envelope(payload="done")

    async def stream(self, env: Envelope, **kwargs):
        if False:
            yield  # pragma: no cover
        return


def _stub_agents():
    a = Agent(name="alpha")
    return [a]


def test_blackboard_state_resets_between_runs() -> None:
    sub = _stub_agents()
    # ``model="claude-opus-4-7"`` is just to satisfy LLMEngine construction
    # (a real provider rule must match in 0.8.0); the engine is replaced
    # immediately below so no provider call is ever made.
    planner = make_blackboard_planner(agents=sub, model="claude-opus-4-7")
    planner.engine = _CapturingMockEngine()  # type: ignore[assignment]

    # Find the closure-held state via the registered tools.
    set_plan_tool = planner._tool_map["set_plan"]
    get_plan_tool = planner._tool_map["get_plan"]

    # Seed the blackboard via the live tool, then run the agent — that should
    # reset state.
    set_plan_tool.func("first reasoning", ["task A", "task B"])
    assert "task A" in get_plan_tool.func()

    asyncio.run(planner.run("hello"))

    # After a run, state is reset; a fresh get_plan reflects that.
    assert "no plan" in get_plan_tool.func() or "(no plan" in get_plan_tool.func()


def test_blackboard_run_does_not_carry_prior_results() -> None:
    sub = _stub_agents()
    planner = make_blackboard_planner(agents=sub, model="claude-opus-4-7")
    planner.engine = _CapturingMockEngine()  # type: ignore[assignment]

    set_plan_tool = planner._tool_map["set_plan"]
    mark_done_tool = planner._tool_map["mark_done"]
    get_plan_tool = planner._tool_map["get_plan"]

    set_plan_tool.func("r", ["t1", "t2"])
    mark_done_tool.func(0, "first done")
    assert "first done" in get_plan_tool.func()

    asyncio.run(planner.run("second task"))
    # Second run starts clean; calling mark_done before set_plan must reject.
    out = mark_done_tool.func(0, "x")
    assert "REJECTED" in out
