"""Unit tests for LazyTool — no API calls, no provider required."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from lazybridge.lazy_tool import (
    LazyTool,
    NormalizedToolSet,
    ToolArgumentValidationError,
)
from lazybridge.core.types import ToolDefinition


# ── Helper functions used as tools ───────────────────────────────────────────

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello {name}"


def scale(x: float) -> float:
    """Scale a value."""
    return x * 2.0


def no_annotations(a, b):
    return a + b


def with_kwargs(a: int, **kwargs) -> int:
    return a


async def async_double(x: int) -> int:
    return x * 2


# ── T1.01 — from_function: default name and description ──────────────────────

def test_from_function_default_name_desc():
    tool = LazyTool.from_function(add)
    assert tool.name == "add"
    assert tool.description == "Add two numbers."


# ── T1.02 — from_function: explicit name/description override ────────────────

def test_from_function_override_name_desc():
    tool = LazyTool.from_function(add, name="my_add", description="Custom desc")
    assert tool.name == "my_add"
    assert tool.description == "Custom desc"


# ── T1.03 — run: int coerced to float ────────────────────────────────────────

def test_run_coercion():
    tool = LazyTool.from_function(scale)
    result = tool.run({"x": 3})      # int 3 → float 3.0
    assert result == 6.0


# ── T1.04 — run: missing required argument → ToolArgumentValidationError ─────

def test_run_missing_argument():
    tool = LazyTool.from_function(add)
    with pytest.raises(ToolArgumentValidationError):
        tool.run({"a": 1})           # b is missing


# ── T1.05 — run: wrong type that can't be coerced → ToolArgumentValidationError

def test_run_bad_type():
    tool = LazyTool.from_function(add)
    with pytest.raises(ToolArgumentValidationError):
        tool.run({"a": "not_an_int", "b": 2})


# ── T1.06 — from_agent: fixed {"task": str} schema, _delegate set ─────────────

def test_from_agent_schema():
    agent = MagicMock()
    agent.name = "my_agent"
    agent.description = "Does stuff"
    tool = LazyTool.from_agent(agent)
    defn = tool.definition()
    assert defn.parameters["properties"] == {"task": {"type": "string"}}
    assert tool._delegate is not None
    assert tool._delegate.agent is agent


# ── T1.07 — specialize: delegate tool with name override updates _compiled ────

def test_specialize_delegate_updates_compiled():
    agent = MagicMock()
    agent.name = "agent"
    agent.description = "desc"
    tool = LazyTool.from_agent(agent)
    specialized = tool.specialize(name="new_name", description="new_desc")
    assert specialized.name == "new_name"
    assert specialized.description == "new_desc"
    assert specialized._compiled is not None
    assert specialized._compiled.name == "new_name"
    assert specialized._compiled.description == "new_desc"
    # Parameters unchanged
    assert specialized._compiled.parameters == tool._compiled.parameters


# ── T1.08 — specialize: function tool resets _compiled ───────────────────────

def test_specialize_function_resets_compiled():
    tool = LazyTool.from_function(add)
    _ = tool.definition()            # cache it
    assert tool._compiled is not None
    specialized = tool.specialize(name="add_v2")
    assert specialized.name == "add_v2"
    assert specialized._compiled is None   # cleared so it rebuilds


# ── T1.09 — definition: result is cached after first call ─────────────────────

def test_definition_cached():
    tool = LazyTool.from_function(add)
    defn1 = tool.definition()
    defn2 = tool.definition()
    assert defn1 is defn2


# ── T1.10 — arun: sync function wrapped as awaitable ─────────────────────────

async def test_arun_sync_function():
    tool = LazyTool.from_function(add)
    result = await tool.arun({"a": 3, "b": 4})
    assert result == 7


# ── T1.11 — arun: async function awaited correctly ───────────────────────────

async def test_arun_async_function():
    tool = LazyTool.from_function(async_double)
    result = await tool.arun({"x": 5})
    assert result == 10


# ── T1.12 — NormalizedToolSet: duplicate name raises ValueError ───────────────

def test_normalized_toolset_duplicate_name():
    tool_a = LazyTool.from_function(add)
    tool_b = LazyTool.from_function(add)   # same name "add"
    with pytest.raises(ValueError, match="Duplicate tool name"):
        NormalizedToolSet.from_list([tool_a, tool_b])


# ── T1.13 — NormalizedToolSet: mix of LazyTool, ToolDefinition, dict ─────────

def test_normalized_toolset_mixed_types():
    lazy = LazyTool.from_function(greet)
    defn = ToolDefinition(
        name="explicit_defn",
        description="A ToolDefinition",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    raw = {
        "name": "raw_dict",
        "description": "A dict tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    ts = NormalizedToolSet.from_list([lazy, defn, raw])
    assert len(ts.definitions) == 3
    assert len(ts.bridges) == 1          # only LazyTool
    assert "greet" in ts.registry
    assert "explicit_defn" not in ts.registry
    assert "raw_dict" not in ts.registry
