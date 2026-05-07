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
    # wrap_tool routes through as_tool() which sets returns_envelope=True
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
    """When signature and hybrid both fail, auto must raise rather than call LLM."""
    import lazybridge.tools as _tools

    call_count = {"llm": 0}

    original = _tools._resolve_auto_mode

    def _patched(func, mode, schema_llm, allow_llm_schema):
        if mode != "auto":
            return mode
        # Simulate both signature and hybrid failing
        errors = [("signature", "no type hints"), ("hybrid", "no doc")]
        if allow_llm_schema or schema_llm is not None:
            call_count["llm"] += 1
            return "llm"
        fn_name = getattr(func, "__name__", repr(func))
        raise ValueError(
            f"Could not build tool schema for {fn_name!r} using signature or hybrid mode.\n"
            f"  signature error: {errors[0][1]}\n"
            f"  hybrid error: {errors[1][1]}\n"
            f"To allow LLM schema generation, pass:\n"
            f"    allow_llm_schema=True"
        )

    monkeypatch.setattr(_tools, "_resolve_auto_mode", _patched)

    def _untyped(x):
        pass

    with pytest.raises(ValueError, match="allow_llm_schema"):
        tool(_untyped, name="fn")

    assert call_count["llm"] == 0


def test_tool_auto_uses_llm_with_opt_in(monkeypatch):
    """With allow_llm_schema=True, auto should select the llm path."""
    import lazybridge.tools as _tools

    call_count = {"llm": 0}

    def _patched(func, mode, schema_llm, allow_llm_schema):
        if mode != "auto":
            return mode
        if allow_llm_schema or schema_llm is not None:
            call_count["llm"] += 1
            return "llm"
        raise ValueError("no opt-in")

    monkeypatch.setattr(_tools, "_resolve_auto_mode", _patched)

    def _untyped(x):
        pass

    # We don't actually build the schema here — just verify mode resolution
    # returns "llm" so the Tool is constructed with that mode.
    from lazybridge.tools import Tool as _Tool

    original_init = _Tool.__init__

    captured_mode = {}

    def _fake_init(self, func, *, name=None, description=None, mode="signature", **kw):
        captured_mode["mode"] = mode
        original_init(self, func, name=name, description=description, mode=mode, **kw)

    monkeypatch.setattr(_Tool, "__init__", _fake_init)

    tool(_untyped, name="fn", allow_llm_schema=True)
    assert captured_mode.get("mode") == "llm"
    assert call_count["llm"] == 1
