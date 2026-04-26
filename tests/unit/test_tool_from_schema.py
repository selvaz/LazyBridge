"""Direct unit tests for ``Tool.from_schema`` and ``build_tool_map``
provider-expansion semantics. Both are core pieces underpinning the MCP
integration but are also useful general-purpose APIs in their own right.
"""

from __future__ import annotations

import asyncio
import warnings

from lazybridge import Agent
from lazybridge.tools import Tool, build_tool_map

# ---------------------------------------------------------------------------
# Tool.from_schema — happy path
# ---------------------------------------------------------------------------


def test_from_schema_carries_name_description_parameters() -> None:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    t = Tool.from_schema(
        name="fs.read_file",
        description="Read a file.",
        parameters=schema,
        func=lambda **kw: f"contents:{kw}",
    )
    d = t.definition()
    assert d.name == "fs.read_file"
    assert d.description == "Read a file."
    assert d.parameters == schema
    assert d.strict is False


def test_from_schema_definition_is_idempotent() -> None:
    """``definition()`` returns the cached ToolDefinition on every call."""
    t = Tool.from_schema(
        name="x",
        description="x",
        parameters={"type": "object", "properties": {}},
        func=lambda: "ok",
    )
    a = t.definition()
    b = t.definition()
    assert a is b


def test_from_schema_strict_flag_propagates() -> None:
    t = Tool.from_schema(
        name="x",
        description="x",
        parameters={"type": "object", "properties": {}},
        func=lambda: None,
        strict=True,
    )
    assert t.definition().strict is True
    assert t.strict is True


def test_from_schema_returns_envelope_flag_propagates() -> None:
    t = Tool.from_schema(
        name="x",
        description="x",
        parameters={"type": "object", "properties": {}},
        func=lambda: None,
        returns_envelope=True,
    )
    assert t.returns_envelope is True


# ---------------------------------------------------------------------------
# Tool.from_schema — empty / minimal schemas accepted verbatim
# ---------------------------------------------------------------------------


def test_from_schema_accepts_empty_object_schema() -> None:
    """A no-arg tool can be wrapped with an empty properties object."""
    t = Tool.from_schema(
        name="now",
        description="Return the current time.",
        parameters={"type": "object", "properties": {}},
        func=lambda: "2026-01-01",
    )
    d = t.definition()
    assert d.parameters["properties"] == {}


def test_from_schema_passes_complex_jsonschema_through_unchanged() -> None:
    """JSON Schema features beyond plain object/properties survive verbatim."""
    schema = {
        "type": "object",
        "properties": {
            "filter": {
                "oneOf": [
                    {"type": "string", "enum": ["all", "open", "closed"]},
                    {"type": "null"},
                ]
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["filter"],
        "additionalProperties": False,
    }
    t = Tool.from_schema(
        name="github.list_issues",
        description="List issues.",
        parameters=schema,
        func=lambda **kw: [],
    )
    assert t.definition().parameters == schema  # unchanged


# ---------------------------------------------------------------------------
# Tool.from_schema — runtime invocation
# ---------------------------------------------------------------------------


def test_from_schema_run_invokes_underlying_func_with_kwargs() -> None:
    captured: list[dict] = []

    def fake(**kwargs):
        captured.append(kwargs)
        return f"done:{kwargs.get('path')}"

    t = Tool.from_schema(
        name="fs.read_file",
        description="Read",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        func=fake,
    )
    out = asyncio.run(t.run(path="/tmp/x"))
    assert out == "done:/tmp/x"
    assert captured == [{"path": "/tmp/x"}]


def test_from_schema_run_supports_async_funcs() -> None:
    async def fake(**kwargs):
        return f"async:{kwargs}"

    t = Tool.from_schema(
        name="x",
        description="x",
        parameters={"type": "object", "properties": {}},
        func=fake,
    )
    out = asyncio.run(t.run())
    assert out.startswith("async:")


# ---------------------------------------------------------------------------
# build_tool_map provider expansion
# ---------------------------------------------------------------------------


class _FakeToolProvider:
    """Minimal tool provider for direct testing (not via MCP)."""

    _is_lazy_tool_provider = True

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools
        self.expansions = 0

    def as_tools(self) -> list[Tool]:
        self.expansions += 1
        return list(self._tools)


def test_build_tool_map_expands_lazy_tool_provider() -> None:
    a = Tool.from_schema("ns.x", "x", {"type": "object", "properties": {}}, lambda: 1)
    b = Tool.from_schema("ns.y", "y", {"type": "object", "properties": {}}, lambda: 2)
    provider = _FakeToolProvider([a, b])

    out = build_tool_map([provider])
    assert sorted(out.keys()) == ["ns.x", "ns.y"]
    assert provider.expansions == 1


def test_build_tool_map_mixes_providers_with_plain_callables() -> None:
    def search(q: str) -> str:
        """Search."""
        return q

    provider = _FakeToolProvider(
        [Tool.from_schema("p.first", "first", {"type": "object", "properties": {}}, lambda: 1)]
    )
    out = build_tool_map([search, provider])
    assert "search" in out
    assert "p.first" in out


def test_build_tool_map_collision_warns_only_once_per_name() -> None:
    """Reproduce the audit fix: a triple-collision should emit ONE warning,
    not two (previously the second + third both warned)."""

    def f1():
        """f1."""
        return 1

    def f2():
        """f2."""
        return 2

    def f3():
        """f3."""
        return 3

    f2.__name__ = "f1"  # collision against the first
    f3.__name__ = "f1"  # second collision against the same name

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_tool_map([f1, f2, f3])

    collisions = [w for w in caught if "Tool name collision" in str(w.message) and "f1" in str(w.message)]
    assert len(collisions) == 1, (
        f"expected exactly one collision warning per name, got {len(collisions)}: "
        f"{[str(w.message) for w in collisions]}"
    )


def test_build_tool_map_empty_provider_contributes_nothing() -> None:
    provider = _FakeToolProvider([])
    out = build_tool_map([provider])
    assert out == {}


def test_agent_constructor_accepts_provider_in_tools_list() -> None:
    a = Tool.from_schema("p.x", "x", {"type": "object", "properties": {}}, lambda: None)
    provider = _FakeToolProvider([a])
    agent = Agent(engine_or_model="claude-opus-4-7", tools=[provider])
    assert "p.x" in agent._tool_map
