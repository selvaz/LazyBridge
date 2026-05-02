"""Unit tests for ``lazybridge.ext.mcp``.

The official MCP SDK is NOT required for these tests — they exercise the
LazyBridge integration via :meth:`MCP.from_transport`, passing a fake
transport that implements the abstract :class:`_Transport` interface.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lazybridge import Agent
from lazybridge.ext.mcp import MCP, MCPServer
from lazybridge.ext.mcp.transports import _Transport
from lazybridge.testing import MockAgent

# ---------------------------------------------------------------------------
# Fake transport — captures call_tool invocations; configurable tool list.
# ---------------------------------------------------------------------------


class FakeTransport(_Transport):
    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        # Use ``is None`` (not ``or``) so the caller can pass ``tools=[]``
        # to model an empty catalogue without falling back to the default.
        self._tools = (
            tools
            if tools is not None
            else [
                {
                    "name": "list_directory",
                    "description": "List the contents of a directory.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
                {
                    "name": "read_file",
                    "description": "Read a file from disk.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
                {
                    "name": "delete_file",
                    "description": "Delete a file from disk.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            ]
        )
        self.connected = False
        self.closed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def connect(self) -> None:
        self.connected = True

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return f"result of {name}({arguments})"

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def test_mcp_module_stability_is_stable() -> None:
    import lazybridge.ext.mcp as mcp_pkg

    assert mcp_pkg.__stability__ == "stable"
    assert mcp_pkg.__lazybridge_min__ == "1.0.0"


def test_mcpserver_is_marked_as_tool_provider() -> None:
    assert MCPServer._is_lazy_tool_provider is True


# ---------------------------------------------------------------------------
# Tool expansion + namespacing
# ---------------------------------------------------------------------------


def test_as_tools_expands_to_namespaced_tools() -> None:
    fs = MCP.from_transport("fs", FakeTransport())
    tools = fs.as_tools()
    names = [t.name for t in tools]
    assert names == ["fs.list_directory", "fs.read_file", "fs.delete_file"]


def test_namespacing_can_be_disabled() -> None:
    fs = MCP.from_transport("fs", FakeTransport(), namespace=False)
    names = [t.name for t in fs.as_tools()]
    assert names == ["list_directory", "read_file", "delete_file"]


def test_namespacing_prefix_can_be_overridden() -> None:
    fs = MCP.from_transport("fs", FakeTransport(), prefix="myfs_")
    names = [t.name for t in fs.as_tools()]
    assert names == ["myfs_list_directory", "myfs_read_file", "myfs_delete_file"]


def test_tools_carry_input_schema_from_mcp() -> None:
    fs = MCP.from_transport("fs", FakeTransport())
    [list_dir, *_] = fs.as_tools()
    d = list_dir.definition()
    assert d.parameters["type"] == "object"
    assert "path" in d.parameters["properties"]
    assert d.parameters.get("required") == ["path"]


def test_as_tools_caches_after_first_call() -> None:
    transport = FakeTransport()
    fs = MCP.from_transport("fs", transport)
    a = fs.as_tools()
    b = fs.as_tools()
    assert a is b  # cached identity


# ---------------------------------------------------------------------------
# Agent integration via build_tool_map expansion
# ---------------------------------------------------------------------------


def test_agent_tools_argument_accepts_mcp_server_directly() -> None:
    agent = Agent(
        engine_or_model=MockAgent.__name__,  # placeholder; we won't run
        tools=[MCP.from_transport("fs", FakeTransport())],
    )
    expected = {"fs.list_directory", "fs.read_file", "fs.delete_file"}
    assert expected.issubset(set(agent._tool_map.keys()))


def test_agent_tools_can_mix_mcp_with_plain_callables() -> None:
    def search(query: str) -> str:
        """Plain function used alongside an MCP server."""
        return f"hit: {query}"

    agent = Agent(
        engine_or_model="claude-opus-4-7",
        tools=[search, MCP.from_transport("fs", FakeTransport())],
    )
    assert "search" in agent._tool_map
    assert "fs.read_file" in agent._tool_map


# ---------------------------------------------------------------------------
# Calls round-trip through the transport
# ---------------------------------------------------------------------------


def test_calling_a_wrapped_tool_dispatches_through_transport() -> None:
    transport = FakeTransport()
    fs = MCP.from_transport("fs", transport)
    [list_dir, *_] = fs.as_tools()
    result = asyncio.run(list_dir.run(path="/tmp"))
    assert result == "result of list_directory({'path': '/tmp'})"
    assert transport.calls == [("list_directory", {"path": "/tmp"})]


# ---------------------------------------------------------------------------
# Allow / deny filtering
# ---------------------------------------------------------------------------


def test_allow_pattern_keeps_only_matching_tools() -> None:
    fs = MCP.from_transport(
        "fs",
        FakeTransport(),
        allow=["fs.list_*", "fs.read_*"],
    )
    names = [t.name for t in fs.as_tools()]
    assert names == ["fs.list_directory", "fs.read_file"]


def test_deny_pattern_removes_dangerous_tools() -> None:
    fs = MCP.from_transport(
        "fs",
        FakeTransport(),
        deny=["fs.delete_*"],
    )
    names = [t.name for t in fs.as_tools()]
    assert names == ["fs.list_directory", "fs.read_file"]


def test_allow_and_deny_compose_allow_then_deny() -> None:
    fs = MCP.from_transport(
        "fs",
        FakeTransport(),
        allow=["fs.*_file"],
        deny=["fs.delete_*"],
    )
    names = [t.name for t in fs.as_tools()]
    assert names == ["fs.read_file"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lazy_connect_on_first_as_tools() -> None:
    transport = FakeTransport()
    fs = MCP.from_transport("fs", transport)
    assert transport.connected is False
    fs.as_tools()
    assert transport.connected is True


def test_async_context_manager_connects_and_closes() -> None:
    async def inner() -> tuple[bool, bool, bool]:
        transport = FakeTransport()
        fs = MCP.from_transport("fs", transport)
        async with fs:
            connected_inside = transport.connected
            assert connected_inside
        return (
            transport.connected,
            connected_inside,
            transport.closed,
        )

    after_connect, inside, after_close = asyncio.run(inner())
    assert inside is True
    assert after_close is True


def test_closed_server_cannot_reconnect() -> None:
    async def inner() -> None:
        transport = FakeTransport()
        fs = MCP.from_transport("fs", transport)
        async with fs:
            pass
        with pytest.raises(RuntimeError, match="closed"):
            await fs.aconnect()

    asyncio.run(inner())


# ---------------------------------------------------------------------------
# MCPServer wraps tool functions with introspectable hints
# ---------------------------------------------------------------------------


def test_wrapped_func_carries_mcp_metadata() -> None:
    fs = MCP.from_transport("fs", FakeTransport())
    [list_dir, *_] = fs.as_tools()
    f = list_dir.func
    assert getattr(f, "__mcp_tool_name__", None) == "list_directory"
    assert getattr(f, "__mcp_server_name__", None) == "fs"


# ---------------------------------------------------------------------------
# Edge cases: error surfacing, empty catalogues, lazy lock.
# ---------------------------------------------------------------------------


class _RaisingTransport(_Transport):
    """Fake transport whose :meth:`connect` always fails — used to verify
    that lazy-connect errors surface at ``Agent(tools=[server])`` time
    (or at ``as_tools()`` if called directly), not at first user query."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def connect(self) -> None:
        raise self._error

    async def list_tools(self) -> list[dict[str, Any]]:
        return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return None

    async def close(self) -> None:
        pass


def test_transport_connect_error_surfaces_at_as_tools() -> None:
    fs = MCP.from_transport("fs", _RaisingTransport(RuntimeError("subprocess failed")))
    with pytest.raises(RuntimeError, match="subprocess failed"):
        fs.as_tools()


def test_transport_connect_error_surfaces_at_agent_construction() -> None:
    """Agent(tools=[server]) calls build_tool_map → as_tools → connect.
    A failing connect should bubble up immediately."""
    fs = MCP.from_transport("fs", _RaisingTransport(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        Agent(engine_or_model="claude-opus-4-7", tools=[fs])


def test_empty_tool_catalogue_yields_no_tools() -> None:
    fs = MCP.from_transport("fs", FakeTransport(tools=[]))
    assert fs.as_tools() == []


def test_allow_filter_to_zero_tools_succeeds_silently() -> None:
    """Filtering down to nothing is a valid (if degenerate) configuration."""
    fs = MCP.from_transport(
        "fs",
        FakeTransport(),
        allow=["fs.nonexistent_*"],  # matches nothing
    )
    assert fs.as_tools() == []


def test_allow_and_deny_intersecting_to_empty_succeeds_silently() -> None:
    fs = MCP.from_transport(
        "fs",
        FakeTransport(),
        allow=["fs.*"],
        deny=["fs.*"],
    )
    assert fs.as_tools() == []


def test_lock_lazy_init_does_not_create_during_construction() -> None:
    """Constructing an MCPServer outside an event loop must not create
    an asyncio.Lock (deprecated / errors on Python ≥3.12)."""
    fs = MCP.from_transport("fs", FakeTransport())
    # The lock attribute must be present but None until first async use.
    assert fs._lock is None


def test_lock_initialised_on_first_async_use() -> None:
    fs = MCP.from_transport("fs", FakeTransport())
    asyncio.run(fs.aconnect())
    assert fs._lock is not None
    asyncio.run(fs.aclose())
