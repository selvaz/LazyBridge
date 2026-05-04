"""Unit tests for the in-box planner factories.

Covers:
  - ``lazybridge.ext.planners.make_planner`` (DAG builder + 5 builder tools)
  - ``lazybridge.ext.planners.make_blackboard_planner`` (flat todo list)

End-to-end behaviour is exercised via :class:`MockAgent` so no provider
credentials are required.
"""

import asyncio

import pytest

from lazybridge.ext.planners import (
    BLACKBOARD_PLANNER_GUIDANCE,
    PLANNER_GUIDANCE,
    PLANNER_VERIFY_PROMPT,
    make_blackboard_planner,
    make_execute_plan_tool,
    make_plan_builder_tools,
    make_planner,
)
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agents() -> dict[str, MockAgent]:
    return {
        "research": MockAgent(
            lambda env: f"FACT[{env.task[:60]}]",
            name="research",
            description="research",
        ),
        "math": MockAgent(
            lambda env: f"NUM({env.text()[:60]})",
            name="math",
            description="math",
        ),
        "writer": MockAgent(
            lambda env: f"PROSE<<{env.text()[:300]}>>",
            name="writer",
            description="writer",
        ),
    }


def _registry() -> dict[str, MockAgent]:
    return _agents()


# ---------------------------------------------------------------------------
# make_planner — wiring + guards
# ---------------------------------------------------------------------------


def test_make_planner_wires_subagents_plus_five_builder_tools() -> None:
    a = _agents()
    planner = make_planner(list(a.values()))
    expected = sorted(
        ["research", "math", "writer", "create_plan", "add_step", "inspect_plan", "run_plan", "discard_plan"]
    )
    assert sorted(planner._tool_map.keys()) == expected


def test_make_planner_rejects_empty_agents_list() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_planner([])


def test_make_planner_rejects_duplicate_agent_names() -> None:
    a = MockAgent("x", name="dup")
    b = MockAgent("y", name="dup")
    with pytest.raises(ValueError, match="unique names"):
        make_planner([a, b])


def test_planner_guidance_present_and_substantial() -> None:
    assert len(PLANNER_GUIDANCE) > 1000
    for k in ["create_plan", "add_step", "run_plan", "from_parallel_all", "Pitfalls"]:
        assert k in PLANNER_GUIDANCE


def test_planner_verify_prompt_present() -> None:
    assert "approved" in PLANNER_VERIFY_PROMPT
    assert "rejected" in PLANNER_VERIFY_PROMPT


# ---------------------------------------------------------------------------
# Builder tools — happy paths
# ---------------------------------------------------------------------------


def test_builder_create_then_run_linear_pipeline() -> None:
    a = _agents()
    create, add, _, run, _ = make_plan_builder_tools(a)
    pid_msg = asyncio.run(create.run(reasoning="r→w pipeline"))
    pid = pid_msg.split("plan_id=")[1].split(" ")[0]
    asyncio.run(add.run(plan_id=pid, name="r", agent="research"))
    asyncio.run(add.run(plan_id=pid, name="w", agent="writer"))
    out = asyncio.run(run.run(plan_id=pid, task="quantum networking"))
    assert "PROSE<<FACT[quantum networking]>>" in out


def test_builder_run_consumes_plan() -> None:
    a = _agents()
    create, add, inspect, run, _ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="x")).split("plan_id=")[1].split(" ")[0]
    asyncio.run(add.run(plan_id=pid, name="r", agent="research"))
    asyncio.run(run.run(plan_id=pid, task="t"))
    # Subsequent inspect must report unknown plan_id.
    out = asyncio.run(inspect.run(plan_id=pid))
    assert out.startswith("REJECTED") and "unknown plan_id" in out


def test_builder_discard_plan() -> None:
    a = _agents()
    create, _, inspect, _, discard = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="abandon me")).split("plan_id=")[1].split(" ")[0]
    out = asyncio.run(discard.run(plan_id=pid))
    assert out.startswith("ok")
    assert asyncio.run(inspect.run(plan_id=pid)).startswith("REJECTED")


# ---------------------------------------------------------------------------
# Builder tools — local validation messages (the LLM's self-correction loop)
# ---------------------------------------------------------------------------


def test_add_step_rejects_unknown_agent_with_available_list() -> None:
    a = _agents()
    create, add, *_ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="x")).split("plan_id=")[1].split(" ")[0]
    out = asyncio.run(add.run(plan_id=pid, name="x", agent="missing"))
    assert out.startswith("REJECTED") and "missing" in out and "Available" in out


def test_add_step_rejects_duplicate_name() -> None:
    a = _agents()
    create, add, *_ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="x")).split("plan_id=")[1].split(" ")[0]
    asyncio.run(add.run(plan_id=pid, name="r", agent="research"))
    out = asyncio.run(add.run(plan_id=pid, name="r", agent="math"))
    assert out.startswith("REJECTED") and "duplicate" in out


def test_add_step_rejects_forward_from_step_reference() -> None:
    a = _agents()
    create, add, *_ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="x")).split("plan_id=")[1].split(" ")[0]
    out = asyncio.run(
        add.run(
            plan_id=pid,
            name="bad",
            agent="writer",
            task_kind="from_step",
            task_step="future",
        )
    )
    assert out.startswith("REJECTED") and "not yet defined" in out


def test_add_step_rejects_literal_without_task_text() -> None:
    a = _agents()
    create, add, *_ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="x")).split("plan_id=")[1].split(" ")[0]
    out = asyncio.run(add.run(plan_id=pid, name="x", agent="research", task_kind="literal"))
    assert out.startswith("REJECTED") and "task_text" in out


def test_add_step_rejects_from_parallel_all_on_non_parallel_target() -> None:
    a = _agents()
    create, add, *_ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="x")).split("plan_id=")[1].split(" ")[0]
    asyncio.run(
        add.run(plan_id=pid, name="serial", agent="research", task_kind="literal", task_text="x")
    )  # parallel=False
    out = asyncio.run(
        add.run(plan_id=pid, name="join", agent="writer", task_kind="from_parallel_all", task_step="serial")
    )
    assert out.startswith("REJECTED") and "parallel=True" in out


def test_run_plan_rejects_empty_plan() -> None:
    a = _agents()
    create, _, _, run, _ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="empty")).split("plan_id=")[1].split(" ")[0]
    out = asyncio.run(run.run(plan_id=pid, task="t"))
    assert out.startswith("REJECTED") and "no steps" in out


# ---------------------------------------------------------------------------
# Builder — N-branch synthesis via from_parallel_all (closes the footgun)
# ---------------------------------------------------------------------------


def test_builder_from_parallel_all_synthesises_all_branches() -> None:
    a = _agents()
    create, add, _, run, _ = make_plan_builder_tools(a)
    pid = asyncio.run(create.run(reasoning="5 lookups + synth")).split("plan_id=")[1].split(" ")[0]
    for n, t in [
        ("hc_meta", "Meta"),
        ("hc_apple", "Apple"),
        ("hc_amazon", "Amazon"),
        ("hc_netflix", "Netflix"),
        ("hc_google", "Google"),
    ]:
        asyncio.run(add.run(plan_id=pid, name=n, agent="research", task_kind="literal", task_text=t, parallel=True))
    asyncio.run(add.run(plan_id=pid, name="report", agent="writer", task_kind="from_parallel_all", task_step="hc_meta"))
    asyncio.run(run.run(plan_id=pid, task="x"))
    synth_in = a["writer"].calls[-1].env_in.text()
    for tag in ["Meta", "Apple", "Amazon", "Netflix", "Google"]:
        assert f"FACT[{tag}]" in synth_in, f"writer didn't see all 5 branches; missing FACT[{tag}] in:\n{synth_in}"


# ---------------------------------------------------------------------------
# Backward-compat alias preserved
# ---------------------------------------------------------------------------


def test_make_execute_plan_tool_alias_returns_builder_tools() -> None:
    a = _agents()
    res = make_execute_plan_tool(a)
    assert isinstance(res, list) and len(res) == 5
    assert {t.name for t in res} == {
        "create_plan",
        "add_step",
        "inspect_plan",
        "run_plan",
        "discard_plan",
    }


# ---------------------------------------------------------------------------
# Blackboard planner — wiring + tools
# ---------------------------------------------------------------------------


def test_make_blackboard_planner_wires_three_tools_plus_subagents() -> None:
    a = _agents()
    planner = make_blackboard_planner(list(a.values()))
    expected = sorted(["research", "math", "writer", "set_plan", "get_plan", "mark_done"])
    assert sorted(planner._tool_map.keys()) == expected


def test_make_blackboard_planner_rejects_empty_agents() -> None:
    with pytest.raises(ValueError, match="empty"):
        make_blackboard_planner([])


def test_make_blackboard_planner_rejects_duplicate_names() -> None:
    a = MockAgent("x", name="dup")
    b = MockAgent("y", name="dup")
    with pytest.raises(ValueError, match="unique names"):
        make_blackboard_planner([a, b])


def test_blackboard_set_then_mark_done_flow() -> None:
    planner = make_blackboard_planner(
        [
            MockAgent("R", name="research", description="r"),
        ]
    )
    set_plan = planner._tool_map["set_plan"]
    get_plan = planner._tool_map["get_plan"]
    mark_done = planner._tool_map["mark_done"]

    out = asyncio.run(set_plan.run(reasoning="three steps", tasks=["A", "B", "C"]))
    assert "[ ] A" in out and "next: 0" in out

    out = asyncio.run(mark_done.run(task_index=0, result_summary="A=done"))
    assert "[x] A" in out and "→ A=done" in out and "next: 1" in out

    asyncio.run(mark_done.run(task_index=1, result_summary="B=done"))
    out = asyncio.run(mark_done.run(task_index=2, result_summary="C=done"))
    assert "all tasks done" in out

    assert "all tasks done" in asyncio.run(get_plan.run())


def test_blackboard_set_plan_rejects_empty_inputs() -> None:
    planner = make_blackboard_planner([MockAgent("x", name="research")])
    set_plan = planner._tool_map["set_plan"]
    assert "REJECTED" in asyncio.run(set_plan.run(reasoning="", tasks=["x"]))
    assert "REJECTED" in asyncio.run(set_plan.run(reasoning="why", tasks=[]))


def test_blackboard_mark_done_rejects_invalid_inputs() -> None:
    planner = make_blackboard_planner([MockAgent("x", name="research")])
    set_plan = planner._tool_map["set_plan"]
    mark_done = planner._tool_map["mark_done"]

    # No plan yet
    assert "REJECTED" in asyncio.run(mark_done.run(task_index=0, result_summary="x"))

    asyncio.run(set_plan.run(reasoning="r", tasks=["A", "B"]))
    # Out of range
    out = asyncio.run(mark_done.run(task_index=99, result_summary="x"))
    assert "REJECTED" in out and "out of range" in out
    # Empty summary
    out = asyncio.run(mark_done.run(task_index=0, result_summary=""))
    assert "REJECTED" in out


def test_blackboard_set_plan_revises_clears_done_state() -> None:
    planner = make_blackboard_planner([MockAgent("x", name="research")])
    set_plan = planner._tool_map["set_plan"]
    mark_done = planner._tool_map["mark_done"]

    asyncio.run(set_plan.run(reasoning="r", tasks=["A", "B"]))
    asyncio.run(mark_done.run(task_index=0, result_summary="A done"))
    out = asyncio.run(set_plan.run(reasoning="changed", tasks=["X", "Y"]))
    assert "[x]" not in out
    assert "[ ] X" in out and "[ ] Y" in out


def test_blackboard_planner_guidance_present_and_substantial() -> None:
    assert len(BLACKBOARD_PLANNER_GUIDANCE) > 500
    for k in ["set_plan", "get_plan", "mark_done", "Workflow", "Pitfalls"]:
        assert k in BLACKBOARD_PLANNER_GUIDANCE
