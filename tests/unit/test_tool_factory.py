"""Tests for the canonical tool() factory and the tools=[agent] composition path.

Covers:
  - tool() wrapping callables, Tools, and Agents
  - _name_explicit flag on Agent
  - Enforcement of explicit names in tools=[agent]
  - mode="auto" schema resolution
  - Backward compatibility (raw callable, .as_tool())
  - Graph schema: callable tool vs sub-agent edge
"""

from __future__ import annotations

import pytest

from lazybridge import Agent, Session, tool
from lazybridge.tools import Tool

# ---------------------------------------------------------------------------
# Minimal fake engine — satisfies Agent.__init__ without hitting a real LLM
# ---------------------------------------------------------------------------


class _FakeEngine:
    model = "fake"

    def _validate(self, tool_map):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fn(query: str) -> str:
    """A simple typed function for schema tests."""
    return query


# ---------------------------------------------------------------------------
# 1. tool() wrapping a plain callable
# ---------------------------------------------------------------------------


def test_tool_factory_wraps_function_with_explicit_name():
    t = tool(_fn, name="search")
    assert isinstance(t, Tool)
    assert t.name == "search"


def test_tool_factory_requires_name_for_function():
    with pytest.raises(ValueError, match="explicit name"):
        tool(_fn)


def test_tool_factory_callable_with_description():
    t = tool(_fn, name="search", description="Search the web.")
    assert t.description == "Search the web."


# ---------------------------------------------------------------------------
# 2. tool() when obj is already a Tool
# ---------------------------------------------------------------------------


def test_tool_factory_returns_existing_tool_without_overrides():
    base = Tool(_fn, name="x")
    assert tool(base) is base


def test_tool_factory_clones_existing_tool_with_name_alias():
    base = Tool(_fn, name="x")
    alias = tool(base, name="y")
    assert alias is not base
    assert alias.name == "y"
    assert base.name == "x"  # original unmodified


def test_tool_factory_clone_preserves_func_and_flags():
    base = Tool(_fn, name="x", returns_envelope=True, strict=True)
    alias = tool(base, name="y")
    assert alias.func is base.func
    assert alias.returns_envelope is True
    assert alias.strict is True


def test_tool_factory_clone_applies_description_override():
    base = Tool(_fn, name="x", description="old")
    cloned = tool(base, description="new")
    assert cloned.description == "new"
    assert base.description == "old"


# ---------------------------------------------------------------------------
# 3. _name_explicit on Agent
# ---------------------------------------------------------------------------


def test_name_explicit_true_when_name_kwarg_given():
    a = Agent(name="research", engine=_FakeEngine())
    assert a._name_explicit is True


def test_name_explicit_false_when_no_name_given():
    # ``_FakeEngine`` has ``model = "fake"``, so the auto-name fallback fires
    # (the agent gets ``name="fake"``) and the explicit-flag stays False.
    a = Agent(engine=_FakeEngine())
    assert a._name_explicit is False


def test_name_explicit_false_when_only_model_string():
    # Agent("model-string") derives name from model, not explicit
    a = Agent("claude-opus-4-7")
    assert a._name_explicit is False


def test_name_explicit_true_with_explicit_kwarg():
    """``ObservabilityConfig`` was deleted in 0.7.9; the canonical way to
    name an agent is the flat ``name=`` kwarg.  This test locks the
    explicit-name flag against the post-0.7.9 surface."""
    a = Agent(engine=_FakeEngine(), name="obs-agent")
    assert a._name_explicit is True
    assert a.name == "obs-agent"


# ---------------------------------------------------------------------------
# 4. tools=[agent] requires explicit name
# ---------------------------------------------------------------------------


def test_direct_agent_tool_requires_explicit_name():
    child = Agent(engine=_FakeEngine())  # no name=, _name_explicit stays False
    with pytest.raises(ValueError, match="explicit name"):
        Agent(name="parent", engine=_FakeEngine(), tools=[child])


def test_direct_agent_tool_uses_agent_name():
    child = Agent(name="research", engine=_FakeEngine())
    parent = Agent(name="parent", engine=_FakeEngine(), tools=[child])
    assert "research" in parent._tool_map


def test_direct_agent_tool_returns_envelope_true():
    child = Agent(name="research", engine=_FakeEngine())
    Agent(name="parent", engine=_FakeEngine(), tools=[child])
    # _wrap_tool routes through as_tool() which sets returns_envelope=True
    # Build independently to verify
    t = tool(child)
    assert t.returns_envelope is True


# ---------------------------------------------------------------------------
# 5. tool() wrapping an Agent
# ---------------------------------------------------------------------------


def test_tool_factory_wraps_agent_with_explicit_name_attr():
    child = Agent(name="research", engine=_FakeEngine())
    t = tool(child)
    assert isinstance(t, Tool)
    assert t.name == "research"
    assert t.returns_envelope is True


def test_tool_factory_wraps_agent_with_explicit_alias():
    # Even an agent without _name_explicit can get an alias via the factory
    child = Agent(engine=_FakeEngine())
    t = tool(child, name="my_alias")
    assert t.name == "my_alias"
    assert t.returns_envelope is True


def test_tool_factory_raises_for_implicit_agent_without_alias():
    child = Agent(engine=_FakeEngine())  # _name_explicit=False
    with pytest.raises(ValueError, match="explicit name"):
        tool(child)


# ---------------------------------------------------------------------------
# 6. Plan accepts direct agent tool (Step 14)
# ---------------------------------------------------------------------------


def test_plan_accepts_direct_agent_tool():
    from lazybridge import Plan, Step

    child = Agent(name="research", engine=_FakeEngine())
    parent = Agent(
        name="pipeline",
        engine=Plan(Step("research")),
        tools=[child],
    )
    assert "research" in parent._tool_map


def test_plan_rejects_agent_without_explicit_name():
    from lazybridge import Plan, Step

    child = Agent(engine=_FakeEngine())
    with pytest.raises(ValueError, match="explicit name"):
        Agent(
            name="pipeline",
            engine=Plan(Step("research")),
            tools=[child],
        )


# ---------------------------------------------------------------------------
# 7. .as_tool() still works for alias (backward compat)
# ---------------------------------------------------------------------------


def test_as_tool_still_works_for_alias():
    from lazybridge import Plan, Step

    child = Agent(name="research", engine=_FakeEngine())
    parent = Agent(
        name="pipeline",
        engine=Plan(Step("deep_research")),
        tools=[child.as_tool("deep_research")],
    )
    assert "deep_research" in parent._tool_map


def test_as_tool_no_name_uses_agent_name():
    child = Agent(name="research", engine=_FakeEngine())
    t = child.as_tool()
    assert t.name == "research"


# ---------------------------------------------------------------------------
# 8. Raw callable backward compatibility
# ---------------------------------------------------------------------------


def test_raw_callable_still_supported_for_backward_compatibility():
    parent = Agent(name="a", engine=_FakeEngine(), tools=[_fn])
    assert _fn.__name__ in parent._tool_map


def test_function_tool_is_canonical_with_explicit_name():
    search = tool(_fn, name="search")
    parent = Agent(name="a", engine=_FakeEngine(), tools=[search])
    assert "search" in parent._tool_map


# ---------------------------------------------------------------------------
# 9. Graph schema: callable tool vs sub-agent edge (Step 12)
# ---------------------------------------------------------------------------


def test_graph_schema_callable_tool_goes_to_tool_node():
    sess = Session()
    search = tool(_fn, name="search")
    Agent(name="research", engine=_FakeEngine(), tools=[search], session=sess)

    graph = sess.graph
    node_ids = {n.id for n in graph.nodes()}
    assert "tool:search" in node_ids


def test_graph_schema_direct_agent_tool_is_agent_node_not_tool_node():
    sess = Session()
    child = Agent(name="research", engine=_FakeEngine())
    Agent(name="pipeline", engine=_FakeEngine(), tools=[child], session=sess)

    graph = sess.graph
    node_ids = {n.id for n in graph.nodes()}
    # Child appears as an agent node
    assert "research" in node_ids
    # No phantom tool:research node (returns_envelope=True → skipped in add_agent loop)
    assert "tool:research" not in node_ids


def test_graph_schema_direct_agent_edge_label_is_agent_name():
    sess = Session()
    child = Agent(name="research", engine=_FakeEngine())
    Agent(name="pipeline", engine=_FakeEngine(), tools=[child], session=sess)

    edges = [(e.from_id, e.to_id, e.label) for e in sess.graph.edges()]
    assert ("pipeline", "research", "research") in edges


# ---------------------------------------------------------------------------
# 10. mode="signature" is the only default; "auto" was removed in 0.7.9
# ---------------------------------------------------------------------------


def test_tool_default_mode_is_signature():
    """``tool()`` and ``Tool()`` both default to ``mode="signature"``.
    The ``"auto"`` ladder (with its silent under-description fallback)
    was removed in 0.7.9 — callers wanting LLM-enriched schemas must
    pass ``mode="hybrid"`` or ``mode="llm"`` explicitly."""
    t = tool(_fn, name="fn")
    assert t.mode == "signature"


def test_tool_factory_rejects_legacy_auto_mode():
    """``mode="auto"`` was removed in 0.7.9 — passing it must surface
    as a typing-time / runtime error rather than silently downgrading."""
    import pytest

    with pytest.raises((TypeError, ValueError)):
        # mode="auto" is no longer accepted; Tool() validates the literal.
        tool(_fn, name="fn", mode="auto")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 12. B1 — graph edge when child already shares the same session
# ---------------------------------------------------------------------------


def test_graph_edge_when_child_already_has_same_session():
    """Canonical pattern: all agents built with session= up front.
    The parent → child edge must be registered even when child.session
    is already the same session object."""
    sess = Session()
    child = Agent(name="research", engine=_FakeEngine(), session=sess)

    Agent(name="pipeline", engine=_FakeEngine(), tools=[child], session=sess)

    edges = [(e.from_id, e.to_id, e.label) for e in sess.graph.edges()]
    assert ("pipeline", "research", "research") in edges


def test_graph_edge_not_stolen_when_child_has_different_session():
    """If the child has a different session, the parent must not steal it."""
    sess_outer = Session()
    sess_inner = Session()
    child = Agent(name="research", engine=_FakeEngine(), session=sess_inner)

    Agent(name="pipeline", engine=_FakeEngine(), tools=[child], session=sess_outer)

    assert child.session is sess_inner  # unchanged


# ---------------------------------------------------------------------------
# 13. P2 — strict sentinel in tool() clone path
# ---------------------------------------------------------------------------


def test_tool_clone_can_disable_strict():
    """tool(base, strict=False) must disable strict even when base.strict=True."""
    base = Tool(_fn, name="x", strict=True)
    cloned = tool(base, strict=False)
    assert cloned.strict is False
    assert base.strict is True  # original unmodified


def test_tool_clone_preserves_strict_when_not_overridden():
    """tool(base) without strict= must preserve the original strict value."""
    base = Tool(_fn, name="x", strict=True)
    cloned = tool(base, name="y")  # override name but not strict
    assert cloned.strict is True


def test_tool_clone_can_enable_strict():
    """tool(base, strict=True) must enable strict on a non-strict base."""
    base = Tool(_fn, name="x", strict=False)
    cloned = tool(base, strict=True)
    assert cloned.strict is True


# ---------------------------------------------------------------------------
# 14. mode="hybrid" / mode="llm" still work — they're now the explicit
#     ways to opt into LLM-driven schema generation (auto is gone).
# ---------------------------------------------------------------------------


def _underdescribed(x):
    # No type hints, no docstring → signature schema has no descriptions.
    pass


def test_explicit_hybrid_mode_calls_schema_llm():
    """``mode="hybrid"`` (signature + LLM-enriched descriptions) is now the
    only way to get LLM enrichment — there is no implicit auto-upgrade."""
    calls: list[str] = []

    def fake_llm(prompt: str) -> dict:
        calls.append(prompt)
        return {
            "description": "Enriched tool description.",
            "param_descriptions": {"x": "The x input parameter."},
        }

    result = tool(_underdescribed, name="fn", mode="hybrid", schema_llm=fake_llm)
    assert result.mode == "hybrid"
    defn = result.definition()
    assert "Enriched" in defn.description
    assert len(calls) == 1


def test_explicit_llm_mode_calls_schema_llm_for_full_schema():
    """``mode="llm"`` builds the full schema from the LLM — no fallback ladder."""

    def fake_llm(prompt: str) -> dict:
        return {
            "name": "fn",
            "description": "LLM-generated full description.",
            "params": {
                "x": {"type": "string", "description": "The x parameter.", "required": True},
            },
        }

    result = tool(_underdescribed, name="fn", mode="llm", schema_llm=fake_llm)
    assert result.mode == "llm"
    defn = result.definition()
    assert "LLM-generated" in defn.description
