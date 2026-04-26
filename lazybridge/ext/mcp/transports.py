"""MCP transports — stdio (subprocess) and Streamable HTTP.

Each transport implements the small interface :class:`_Transport`. The
public :class:`MCP` factory in :mod:`lazybridge.ext.mcp.server` builds
transports lazily and only imports the official ``mcp`` SDK when a real
server is constructed — so importing :mod:`lazybridge.ext.mcp` itself is
cheap and never fails.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class _Transport(ABC):
    """Abstract MCP transport.  Sub-classes implement the JSON-RPC surface
    LazyBridge needs (initialise + list-tools + call-tool + close)."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's tool catalogue.

        Each entry must be a dict with at least:

        - ``name``        — str
        - ``description`` — str
        - ``inputSchema`` — JSON Schema dict (object type with ``properties``)
        """

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...

    @abstractmethod
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# stdio: subprocess via the official ``mcp`` SDK.
# ---------------------------------------------------------------------------


class StdioTransport(_Transport):
    """Spawn an MCP server as a subprocess and speak JSON-RPC over stdio."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        self._args = args or []
        self._env = env
        self._session: Any | None = None
        self._stack: Any | None = None  # contextlib.AsyncExitStack

    async def connect(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:  # pragma: no cover — exercised only without [mcp]
            raise ImportError(
                "lazybridge.ext.mcp.MCP.stdio requires the official MCP SDK. "
                "Install with: pip install lazybridge[mcp]"
            ) from e
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def list_tools(self) -> list[dict[str, Any]]:
        assert self._session is not None, "list_tools called before connect"
        result = await self._session.list_tools()
        out: list[dict[str, Any]] = []
        for t in result.tools:
            out.append({
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
            })
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._session is not None, "call_tool called before connect"
        result = await self._session.call_tool(name, arguments=arguments)
        # MCP returns a list of content blocks; flatten text content into a string.
        return _extract_text(result)

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.__aexit__(None, None, None)
            finally:
                self._stack = None
                self._session = None


# ---------------------------------------------------------------------------
# Streamable HTTP — same SDK, different transport.
# ---------------------------------------------------------------------------


class HttpTransport(_Transport):
    """Connect to an MCP server over Streamable HTTP."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        self._url = url
        self._headers = headers or {}
        self._session: Any | None = None
        self._stack: Any | None = None

    async def connect(self) -> None:
        if self._session is not None:
            return
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "lazybridge.ext.mcp.MCP.http requires the official MCP SDK. "
                "Install with: pip install lazybridge[mcp]"
            ) from e
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(self._url, headers=self._headers)
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def list_tools(self) -> list[dict[str, Any]]:
        assert self._session is not None, "list_tools called before connect"
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {"type": "object", "properties": {}},
            }
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        assert self._session is not None, "call_tool called before connect"
        result = await self._session.call_tool(name, arguments=arguments)
        return _extract_text(result)

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.__aexit__(None, None, None)
            finally:
                self._stack = None
                self._session = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _extract_text(result: Any) -> str:
    """Pull text out of an MCP CallToolResult; fall back to repr for non-text."""
    content = getattr(result, "content", None)
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(repr(block))
    return "\n".join(parts)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine to completion from a sync context.

    Handles the case where the caller is already inside a running event
    loop by hopping to a worker thread (mirrors ``Tool.run_sync``).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
