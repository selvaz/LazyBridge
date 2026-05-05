"""Wave: ergonomic ext-side factories — symmetric counterparts of
``Agent.from_<kind>`` for engines that live in :mod:`lazybridge.ext`.

Per the core/ext boundary policy (``docs/guides/core-vs-ext.md``) the
``Agent`` class must not import from ``lazybridge.ext.*``, even via
lazy imports.  So factories that build Agents around ext-only engines
live module-side in their respective ext packages:

* :func:`lazybridge.ext.hil.supervisor_agent`
* :func:`lazybridge.ext.hil.human_agent`
* :func:`lazybridge.ext.planners.orchestrator_agent`  (canonical name)
* :func:`lazybridge.ext.planners.blackboard_orchestrator_agent` (canonical)

Backward-compat aliases:
* ``make_planner`` → ``orchestrator_agent`` (same callable)
* ``make_blackboard_planner`` → ``blackboard_orchestrator_agent``

This module pins the public surface — every name above must remain
importable, every alias must point to the canonical function, and
the kwargs must flow through to the unified Agent constructor.
"""

from __future__ import annotations

import pytest

from lazybridge import Agent, Memory, Session

# ---------------------------------------------------------------------------
# ext.planners — canonical names + backward-compat aliases
# ---------------------------------------------------------------------------


def test_planners_canonical_and_alias_are_identical_callables():
    from lazybridge.ext.planners import (
        blackboard_orchestrator_agent,
        make_blackboard_planner,
        make_planner,
        orchestrator_agent,
    )

    # Aliases are the same callable — not a thin wrapper that doubles
    # the call cost or drifts from the canonical signature.
    assert orchestrator_agent is make_planner
    assert blackboard_orchestrator_agent is make_blackboard_planner


def test_planners_all_documented_symbols_importable():
    """Public surface — every documented name resolves."""
    from lazybridge.ext.planners import (
        BLACKBOARD_PLANNER_GUIDANCE,
        PLANNER_GUIDANCE,
        PLANNER_VERIFY_PROMPT,
        PlanSpec,
        StepSpec,
        blackboard_orchestrator_agent,
        make_blackboard_planner,
        make_execute_plan_tool,
        make_plan_builder_tools,
        make_planner,
        orchestrator_agent,
    )

    # Sanity binds — silence linter on the import-only test.
    for sym in (
        orchestrator_agent,
        blackboard_orchestrator_agent,
        make_planner,
        make_blackboard_planner,
        make_plan_builder_tools,
        make_execute_plan_tool,
        PlanSpec,
        StepSpec,
        PLANNER_GUIDANCE,
        PLANNER_VERIFY_PROMPT,
        BLACKBOARD_PLANNER_GUIDANCE,
    ):
        assert sym is not None


# ---------------------------------------------------------------------------
# ext.hil — module-level factories
# ---------------------------------------------------------------------------


def test_hil_factories_importable():
    from lazybridge.ext.hil import (
        HumanEngine,
        SupervisorEngine,
        human_agent,
        supervisor_agent,
    )

    assert callable(supervisor_agent)
    assert callable(human_agent)
    assert HumanEngine is not None
    assert SupervisorEngine is not None


def test_supervisor_agent_returns_agent_with_supervisor_engine():
    from lazybridge.ext.hil import SupervisorEngine, supervisor_agent

    a = supervisor_agent(tools=[], agents=[])
    assert isinstance(a, Agent)
    assert isinstance(a.engine, SupervisorEngine)


def test_supervisor_agent_threads_engine_kwargs():
    from lazybridge.ext.hil import supervisor_agent

    a = supervisor_agent(
        tools=[],
        agents=[],
        timeout=30.0,
        default="continue",
    )
    assert a.engine.timeout == 30.0
    assert a.engine.default == "continue"


def test_supervisor_agent_threads_agent_kwargs():
    """The unified-surface invariant: name= / session= / memory= reach
    the constructed Agent regardless of which factory built it."""
    from lazybridge.ext.hil import supervisor_agent

    sess = Session()
    mem = Memory()
    a = supervisor_agent(
        tools=[],
        agents=[],
        name="ops-supervisor",
        session=sess,
        memory=mem,
    )
    assert a.name == "ops-supervisor"
    assert a.session is sess
    assert a.memory is mem


def test_human_agent_returns_agent_with_human_engine():
    from lazybridge.ext.hil import HumanEngine, human_agent

    a = human_agent(timeout=10.0, default="approve")
    assert isinstance(a, Agent)
    assert isinstance(a.engine, HumanEngine)
    assert a.engine.timeout == 10.0
    assert a.engine.default == "approve"


def test_human_agent_threads_agent_kwargs():
    from lazybridge.ext.hil import human_agent

    sess = Session()
    a = human_agent(timeout=5.0, name="approver", session=sess)
    assert a.name == "approver"
    assert a.session is sess


# ---------------------------------------------------------------------------
# Composition with from_engine — both routes yield equivalent Agents
# ---------------------------------------------------------------------------


def test_supervisor_via_factory_and_from_engine_match():
    """Both construction paths yield Agents with the same engine type."""
    from lazybridge.ext.hil import SupervisorEngine, supervisor_agent

    by_factory = supervisor_agent(tools=[], agents=[])
    by_engine = Agent.from_engine(SupervisorEngine(tools=[], agents=[]))

    assert type(by_factory.engine) is type(by_engine.engine)


def test_human_via_factory_and_from_engine_match():
    from lazybridge.ext.hil import HumanEngine, human_agent

    by_factory = human_agent(timeout=1.0)
    by_engine = Agent.from_engine(HumanEngine(timeout=1.0))

    assert type(by_factory.engine) is type(by_engine.engine)


# ---------------------------------------------------------------------------
# Boundary check — ext.hil factories never appear on Agent core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["from_supervisor", "from_human", "from_planner", "from_blackboard_planner"])
def test_ext_factories_intentionally_not_on_agent_core(name):
    """These factories live on ext (not Agent core) by design — see
    ``docs/guides/core-vs-ext.md``.  This test pins that decision so a
    future patch can't silently re-add them and re-introduce the
    boundary violation."""
    assert not hasattr(Agent, name), (
        f"Agent.{name} should NOT exist on the core Agent class.  "
        f"Use the ext-side factory in lazybridge.ext.hil / "
        f"lazybridge.ext.planners or wrap the engine via "
        f"Agent.from_engine(...)."
    )


# ---------------------------------------------------------------------------
# Sanity: orchestrator_agent + blackboard_orchestrator_agent build Agents
# ---------------------------------------------------------------------------


def test_orchestrator_agent_builds_agent():
    """A planner-LLM agent over a single sub-agent is a valid Agent
    instance — the factory just configures the right tools list."""
    from lazybridge.ext.planners import orchestrator_agent
    from lazybridge.testing import MockAgent

    sub = MockAgent("sub-output", name="sub")
    a = orchestrator_agent(agents=[sub])
    assert isinstance(a, Agent)
    # Sub-agent must appear in the planner's tools list.
    tool_names = {t.name for t in a._tool_map.values()}
    assert "sub" in tool_names


def test_blackboard_orchestrator_agent_builds_agent():
    from lazybridge.ext.planners import blackboard_orchestrator_agent
    from lazybridge.testing import MockAgent

    sub = MockAgent("sub-output", name="sub")
    a = blackboard_orchestrator_agent(agents=[sub])
    assert isinstance(a, Agent)


def test_orchestrator_rejects_empty_agents_list():
    from lazybridge.ext.planners import orchestrator_agent

    with pytest.raises(ValueError, match="agents list must not be empty"):
        orchestrator_agent(agents=[])


def test_blackboard_orchestrator_rejects_empty_agents_list():
    from lazybridge.ext.planners import blackboard_orchestrator_agent

    with pytest.raises(ValueError, match="agents list must not be empty"):
        blackboard_orchestrator_agent(agents=[])
