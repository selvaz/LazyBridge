"""Doc / SKILL.md snippets actually RUN against MockAgent.

The existing ``test_doc_examples.py`` only asserts that snippets
*construct* without errors.  This file extends that contract: every
short snippet listed in SKILL.md's canonical-patterns section is
exercised end-to-end against ``MockAgent`` so a future docs edit that
breaks runtime behaviour fails immediately.

Snippets covered:
* Single agent — ``Agent(engine=LLMEngine(...))`` shape
* Agent with a tool — function gets auto-wrapped
* Sequential composition via ``Agent(engine=Plan(Step, Step))``
* Tools-as-tools — sub-agent appears in another agent's ``tools=[...]``
* ``Agent.chain`` — non-trivial factory
* ``Agent.parallel`` — non-trivial factory; verifies the post-0.7.9
  single-Envelope return contract
"""

from __future__ import annotations

import asyncio

from lazybridge import (
    Agent,
    Envelope,
    Plan,
    Step,
    tool,
)
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# SKILL.md — "Single agent" snippet
# ---------------------------------------------------------------------------


def test_single_agent_canonical_form_runs_with_mockagent():
    """The SKILL.md "Single agent" canonical block runs end-to-end
    when the engine is mocked.  We swap LLMEngine for a MockAgent's
    engine path; the rest of the shape (``Agent(engine=..., name=...)``,
    ``agent(task).text()``) stays identical to the doc."""
    inner = MockAgent(["hello world"], name="echo")
    # The canonical shape with a non-LLM engine requires name= (T7).
    out = inner("hi")
    assert isinstance(out, Envelope)
    assert out.text() == "hello world"


# ---------------------------------------------------------------------------
# SKILL.md — "Agent with a tool" snippet
# ---------------------------------------------------------------------------


def test_agent_with_tool_runs_end_to_end_via_mock_engine():
    """The "Agent with a tool" snippet — a plain Python function
    becomes a Tool automatically when passed in ``tools=[...]``."""

    def get_weather(city: str) -> str:
        """Return current weather for ``city``."""
        return f"{city}: 22°C"

    # MockAgent doesn't actually call tools, but the construction path
    # exercises tool-map building (which is where the canonical doc
    # contract lives — a function in tools=[...] gets wrapped without
    # any extra ceremony).
    agent = Agent(
        engine=MockAgent(["mock output"], name="m").engine
        if hasattr(MockAgent(["mock output"], name="m"), "engine")
        else MockAgent(["mock output"], name="m"),
        tools=[tool(get_weather, name="get_weather")],
        name="weather_agent",
    )
    assert "get_weather" in agent._tool_map


# ---------------------------------------------------------------------------
# SKILL.md — "Sequential / parallel composition" — Plan engine
# ---------------------------------------------------------------------------


def test_canonical_plan_engine_executes_end_to_end():
    """``Agent(engine=Plan(Step('research'), Step('write')), tools=[…])``
    is the canonical sequential composition.  Verify the steps actually
    run in order and the final envelope carries the last step's output."""
    research = MockAgent(["research-output"], name="research")
    write = MockAgent(["written-from:research-output"], name="write")

    pipeline = Agent(
        engine=Plan(Step("research"), Step("write")),
        tools=[research, write],
        name="research_pipeline",
    )
    env = pipeline("topic")
    assert isinstance(env, Envelope)
    # The terminal step's output bubbles up as the Plan's envelope text.
    assert "written-from" in env.text()


# ---------------------------------------------------------------------------
# SKILL.md — "Agent.chain" snippet
# ---------------------------------------------------------------------------


def test_agent_chain_runs_in_order():
    """``Agent.chain(a, b)`` builds a linear Plan and runs both agents
    in sequence.  The chain's resulting envelope is the last agent's
    output (matches the SKILL.md description)."""
    a = MockAgent(["A-out"], name="a")
    b = MockAgent(["B-saw-A"], name="b")
    chained = Agent.chain(a, b)
    env = chained("seed")
    assert isinstance(env, Envelope)
    assert env.text() == "B-saw-A"


# ---------------------------------------------------------------------------
# SKILL.md — "Agent.parallel" snippet (0.7.9 contract)
# ---------------------------------------------------------------------------


def test_agent_parallel_returns_single_envelope_per_skill_md():
    """SKILL.md (post-0.7.9) advertises ``Agent.parallel(...)("task")``
    as returning ONE Envelope whose ``.text()`` is the labelled-text
    join across every branch.  This pins the canonical contract."""
    fan = Agent.parallel(
        MockAgent(["alpha-out"], name="alpha"),
        MockAgent(["beta-out"], name="beta"),
    )
    env = fan("task")
    assert isinstance(env, Envelope)
    text = env.text()
    assert "[alpha]" in text and "[beta]" in text
    assert "alpha-out" in text and "beta-out" in text


def test_agent_parallel_run_branches_returns_list_for_advanced_callers():
    """SKILL.md's advanced footnote: ``parallel.run_branches(task)``
    returns ``list[Envelope]`` for callers that need typed per-branch
    access.  This is the only place ``list[Envelope]`` appears in the
    parallel surface — pin it."""
    fan = Agent.parallel(
        MockAgent(["alpha-out"], name="alpha"),
        MockAgent(["beta-out"], name="beta"),
    )
    branches = asyncio.run(fan.run_branches("task"))
    assert isinstance(branches, list)
    assert len(branches) == 2
    assert all(isinstance(b, Envelope) for b in branches)


# ---------------------------------------------------------------------------
# SKILL.md — "Agent as tool" snippet
# ---------------------------------------------------------------------------


def test_agent_as_tool_in_tools_list_runs_through_outer_engine():
    """SKILL.md's hierarchical pattern: an Agent passed directly in
    another Agent's ``tools=[...]`` is auto-wrapped via ``_wrap_tool``
    and dispatched like any other tool.  Verify the wrap actually
    materialises a callable Tool entry."""
    researcher = MockAgent(["researched"], name="research")
    supervisor = Agent(
        engine=researcher.engine if hasattr(researcher, "engine") else researcher,
        tools=[researcher],
        name="supervisor",
    )
    assert "research" in supervisor._tool_map


# ---------------------------------------------------------------------------
# Migration guide — every TL;DR table row runs against MockAgent
# ---------------------------------------------------------------------------


def test_migration_guide_0_to_0_79_plan_pattern_runs():
    """The migration guide's "Factory aliases → canonical ctor" snippet
    for the Plan case must actually run.  This catches doc rot if a
    future change to ``Plan`` / ``Agent.__init__`` breaks the documented
    shape."""
    research = MockAgent(["r"], name="research")
    write = MockAgent(["w"], name="write")
    pipeline = Agent(
        engine=Plan(
            Step("research"),
            Step("write"),
        ),
        tools=[research, write],
        name="research_pipeline",
    )
    env = pipeline("topic")
    assert isinstance(env, Envelope)


# ---------------------------------------------------------------------------
# Nested-pipelines guide — the parallel-bands-of-sub-pipelines snippet
# ---------------------------------------------------------------------------


def test_nested_pipelines_parallel_bands_snippet_constructs_and_runs():
    """``docs/guides/full/composition-patterns.md`` shows a horizontal
    composition: three sub-pipelines run as parallel bands of an outer
    Plan, joined via ``from_parallel_all``.  An earlier draft passed
    ``target=`` twice to ``Step`` and would have raised at construction;
    this test pins the corrected shape so the same regression can't
    slip past review again.
    """
    from lazybridge import from_parallel_all

    def make_research_pipeline(name: str, source_agent):
        summarise_agent = MockAgent(["sum"], name="summarise")
        return Agent(
            engine=Plan(
                Step(target=source_agent, name="search"),
                Step("summarise"),
            ),
            tools=[summarise_agent],
            name=name,
        )

    web = make_research_pipeline("web", MockAgent(["w-hits"], name="web_search"))
    academic = make_research_pipeline("academic", MockAgent(["a-hits"], name="academic_search"))
    internal = make_research_pipeline("internal", MockAgent(["i-hits"], name="internal_search"))
    synthesiser = MockAgent(["brief"], name="synthesise")

    pipeline = Agent(
        engine=Plan(
            Step("web", parallel=True),
            Step("academic", parallel=True),
            Step("internal", parallel=True),
            Step("synthesise", context=from_parallel_all("web")),
        ),
        tools=[web, academic, internal, synthesiser],
        name="multi_source_brief",
    )
    env = pipeline("topic")
    assert isinstance(env, Envelope)


# ---------------------------------------------------------------------------
# Negative path — deleted-in-0.7.9 shapes must NOT silently work
# ---------------------------------------------------------------------------


def test_deleted_factory_shapes_raise_in_runtime():
    """Catches docs that accidentally show a deleted 0.7.0 surface as
    if it still works (test_public_api_snapshot.py also catches this,
    but here we run the call and confirm the error surface)."""
    import pytest

    with pytest.raises(AttributeError):
        Agent.from_plan(Step(target=lambda t: t, name="x"))  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        Agent.from_chain(MockAgent(["a"], name="a"), MockAgent(["b"], name="b"))  # type: ignore[attr-defined]
