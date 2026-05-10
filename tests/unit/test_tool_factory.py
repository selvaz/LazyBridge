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
    a = Agent(engine=_FakeEngine())
    assert a._name_explicit is False


def test_name_explicit_false_when_only_model_string():
    # Agent("model-string") derives name from model, not explicit
    a = Agent("claude-opus-4-7")
    assert a._name_explicit is False


def test_name_explicit_true_with_observability_config():
    from lazybridge.core.types import ObservabilityConfig

    obs = ObservabilityConfig(name="obs-agent")
    a = Agent(engine=_FakeEngine(), observability=obs)
    assert a._name_explicit is True


# ---------------------------------------------------------------------------
# 4. tools=[agent] requires explicit name
# ---------------------------------------------------------------------------


def test_direct_agent_tool_requires_explicit_name():
    child = Agent(engine=_FakeEngine())  # no name=
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
# 10. mode="auto" schema resolution
# ---------------------------------------------------------------------------


def test_tool_auto_signature_success():
    t = tool(_fn, name="fn", mode="auto")
    assert t.name == "fn"
    assert t.mode == "signature"


def test_tool_auto_does_not_use_llm_without_opt_in(monkeypatch):
    """_resolve_auto_tool must not invoke LLM paths when allow_llm_schema=False
    and schema_llm is None — even if signature schema is poor quality."""
    import lazybridge.tools as _tools

    llm_called = {"count": 0}

    def _patched(func, name, description, schema_llm, strict, allow_llm_schema):
        # Simulate signature success but enrichment needed, no LLM opt-in.
        assert not allow_llm_schema, "LLM used without opt-in"
        assert schema_llm is None, "schema_llm leaked through"
        # Return a minimal sig tool (no LLM call)
        return _tools.Tool(func, name=name, description=description, mode="signature")

    monkeypatch.setattr(_tools, "_resolve_auto_tool", _patched)

    def _untyped(x):
        pass

    t = tool(_untyped, name="fn")
    assert llm_called["count"] == 0
    assert t.name == "fn"


def test_tool_auto_uses_llm_with_opt_in(monkeypatch):
    """With allow_llm_schema=True, _resolve_auto_tool receives the opt-in flag."""
    import lazybridge.tools as _tools

    received = {}

    def _patched(func, name, description, schema_llm, strict, allow_llm_schema):
        received["allow_llm_schema"] = allow_llm_schema
        # Return sig_tool to avoid needing a real schema_llm engine
        return _tools.Tool(func, name=name, description=description, mode="signature")

    monkeypatch.setattr(_tools, "_resolve_auto_tool", _patched)

    def _untyped(x):
        pass

    tool(_untyped, name="fn", allow_llm_schema=True)
    assert received.get("allow_llm_schema") is True


# ---------------------------------------------------------------------------
# 11. _schema_needs_enrichment helper
# ---------------------------------------------------------------------------


def test_schema_needs_enrichment_false_when_fully_described():
    """A typed, documented function should not need enrichment."""
    from lazybridge.core.types import ToolDefinition
    from lazybridge.tools import _schema_needs_enrichment

    defn = ToolDefinition(
        name="search",
        description="Search the web for information.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
            },
            "required": ["query"],
        },
    )
    assert _schema_needs_enrichment(defn) is False


def test_schema_needs_enrichment_true_when_param_lacks_description():
    from lazybridge.core.types import ToolDefinition
    from lazybridge.tools import _schema_needs_enrichment

    defn = ToolDefinition(
        name="search",
        description="Search the web.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},  # no description
            },
        },
    )
    assert _schema_needs_enrichment(defn) is True


def test_schema_needs_enrichment_true_when_no_tool_description():
    from lazybridge.core.types import ToolDefinition
    from lazybridge.tools import _schema_needs_enrichment

    defn = ToolDefinition(
        name="search",
        description="",
        parameters={"type": "object", "properties": {}},
    )
    assert _schema_needs_enrichment(defn) is True


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
# 14. mode="auto" end-to-end with fake schema_llm
# ---------------------------------------------------------------------------


def _underdescribed(x):
    # No type hints, no docstring → signature schema has no descriptions.
    pass


def test_auto_with_schema_llm_selects_hybrid_for_underdescribed_schema():
    """When a function has no docstring/types and schema_llm is provided,
    _resolve_auto_tool must pick hybrid (not llm, not plain signature)."""
    from lazybridge.tools import _resolve_auto_tool

    calls: list[str] = []

    def fake_llm(prompt: str) -> dict:
        calls.append(prompt)
        # _LLMEnrichment shape expected by hybrid mode
        return {
            "description": "Enriched tool description.",
            "param_descriptions": {"x": "The x input parameter."},
        }

    result = _resolve_auto_tool(
        _underdescribed,
        name="fn",
        description=None,
        schema_llm=fake_llm,
        strict=False,
        allow_llm_schema=False,
    )

    assert result.mode == "hybrid"
    # fake_llm was called exactly once (hybrid enrichment call)
    assert len(calls) == 1
    defn = result.definition()
    assert "Enriched" in defn.description


def test_auto_with_allow_llm_schema_selects_llm_when_hybrid_fails():
    """When hybrid fails (LLM raises for the enrichment call) and
    allow_llm_schema=True, _resolve_auto_tool must fall back to llm mode."""
    from lazybridge.tools import _resolve_auto_tool

    call_count = [0]

    def fake_llm(prompt: str) -> dict:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: hybrid enrichment — simulate failure
            raise RuntimeError("hybrid LLM unavailable")
        # Second call: full LLM schema — _LLMToolSchema shape
        return {
            "name": "fn",
            "description": "LLM-generated full description.",
            "params": {
                "x": {"type": "string", "description": "The x parameter.", "required": True},
            },
        }

    result = _resolve_auto_tool(
        _underdescribed,
        name="fn",
        description=None,
        schema_llm=fake_llm,
        strict=False,
        allow_llm_schema=True,
    )

    assert result.mode == "llm"
    assert call_count[0] == 2  # hybrid attempt + llm attempt
    defn = result.definition()
    assert "LLM-generated" in defn.description
