"""``MCP`` and ``MCPServer`` — the public surface of the MCP integration.

An :class:`MCPServer` is a *tool provider*: when added to
``Agent(tools=[...])`` it expands into a list of :class:`lazybridge.Tool`
objects, one per tool the MCP server exposes. The expansion is lazy —
the underlying transport connects on the first ``as_tools()`` call.

Public factories on :class:`MCP`:

- :meth:`MCP.stdio` — spawn an MCP server as a subprocess and speak
  JSON-RPC over its stdio. Common pattern for npx-launched servers
  like ``@modelcontextprotocol/server-filesystem``.
- :meth:`MCP.http` — connect to an MCP server over Streamable HTTP.
- :meth:`MCP.from_transport` — bring-your-own transport, useful for
  in-process fakes in tests.
"""

from __future__ import annotations

import asyncio
import fnmatch
import time
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from lazybridge.tools import Tool

if TYPE_CHECKING:
    from lazybridge.ext.mcp.transports import _Transport


class MCPServer:
    """A tool provider backed by an MCP server.

    Add it directly to ``Agent(tools=[...])``; the framework calls
    :meth:`as_tools` to expand it into individual :class:`Tool` entries.
    Tool names are namespaced as ``"<server-name>.<mcp-tool-name>"`` by
    default; pass ``namespace=False`` to keep the raw names, or
    ``prefix="..."`` to override.

    The transport connects lazily on first :meth:`as_tools`. For explicit
    cleanup, use the server as an async context manager::

        async with MCP.stdio("fs", command="...", args=[...]) as fs:
            agent = Agent("claude-opus-4-7", tools=[fs])
            await agent.run("...")

    Without that, the transport stays open for the process lifetime; the
    underlying subprocess is normally cleaned up when the parent exits.

    **Closure is terminal.** Once :meth:`aclose` (or the ``async with``
    block) finishes, the server is single-shot: a subsequent
    :meth:`aconnect` / :meth:`as_tools` raises ``RuntimeError``.
    Construct a new ``MCPServer`` if you need to re-use the same
    transport configuration.
    """

    _is_lazy_tool_provider = True

    #: Default lifetime of the discovered-tools cache.  After expiry the
    #: next ``alist_tools()`` call re-fetches from the upstream MCP
    #: server, picking up any tools added or removed since the last
    #: discovery (audit H-E).  ``None`` disables expiry — the
    #: pre-1.0.x behaviour.
    _DEFAULT_CACHE_TTL = 60.0

    def __init__(
        self,
        name: str,
        transport: _Transport,
        *,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = _DEFAULT_CACHE_TTL,
    ) -> None:
        self.name = name
        self._transport = transport
        self._namespace = namespace
        if prefix is not None:
            self._prefix = prefix
        else:
            self._prefix = f"{name}." if namespace else ""
        self._allow = list(allow) if allow else None
        self._deny = list(deny) if deny else None

        if cache_tools_ttl is not None and cache_tools_ttl <= 0:
            raise ValueError(f"cache_tools_ttl must be > 0 or None, got {cache_tools_ttl!r}")
        self._cache_ttl: float | None = cache_tools_ttl
        self._tools_cache: list[Tool] | None = None
        self._tools_cache_ts: float = 0.0
        self._connected = False
        self._closed = False
        # Lazy-init the asyncio.Lock on first async use.  Constructing it
        # inside ``__init__`` (a sync context) couples to whatever event
        # loop happens to be running at instantiation time and warns /
        # raises on Python ≥3.12 when there is none.  Deferring is safe
        # because the lock only ever guards async coroutines.
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # --- lifecycle -----------------------------------------------------

    async def aconnect(self) -> None:
        """Connect the underlying transport. Idempotent."""
        async with self._get_lock():
            if not self._connected:
                if self._closed:
                    raise RuntimeError(f"MCPServer {self.name!r} is closed and cannot be reused")
                await self._transport.connect()
                self._connected = True

    async def alist_tools(self) -> list[Tool]:
        """Discover and wrap the server's tools.

        Cached for ``cache_tools_ttl`` seconds (default 60 s).  Once the
        cache expires the next call re-fetches from the upstream
        transport so an MCP server that hot-loads or unloads tools is
        eventually reflected in the agent's tool list (audit H-E).
        Pass ``cache_tools_ttl=None`` to disable expiry entirely (the
        pre-1.0.x behaviour) and :meth:`invalidate_tools_cache` to flush
        explicitly.
        """
        await self.aconnect()
        now = time.monotonic()
        if self._tools_cache is not None and (
            self._cache_ttl is None or (now - self._tools_cache_ts) < self._cache_ttl
        ):
            return self._tools_cache
        mcp_tools = await self._transport.list_tools()
        wrapped = [self._wrap_tool(t) for t in mcp_tools]
        self._tools_cache = self._filter(wrapped)
        self._tools_cache_ts = now
        return self._tools_cache

    def invalidate_tools_cache(self) -> None:
        """Drop the cached tool list so the next call re-fetches.

        Use this when an out-of-band signal tells you the MCP server's
        tool registry has changed (plugin install / uninstall, hot
        reload).  No-op when nothing is cached yet.
        """
        self._tools_cache = None
        self._tools_cache_ts = 0.0

    async def aclose(self) -> None:
        """Close the underlying transport. Idempotent."""
        async with self._get_lock():
            if self._connected and not self._closed:
                try:
                    await self._transport.close()
                finally:
                    self._connected = False
                    self._closed = True

    async def __aenter__(self) -> MCPServer:
        await self.aconnect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    # --- sync façade for build_tool_map -------------------------------

    def as_tools(self) -> list[Tool]:
        """Sync wrapper around :meth:`alist_tools`. Called by ``build_tool_map``.

        Triggers a lazy connect on first use. If the call happens inside
        an already-running event loop, the work is dispatched to a worker
        thread (mirrors :meth:`lazybridge.Tool.run_sync`).
        """
        from lazybridge.ext.mcp.transports import _run_sync

        return _run_sync(self.alist_tools())

    # --- internal helpers ---------------------------------------------

    def _wrap_tool(self, mcp_tool: dict[str, Any]) -> Tool:
        local_name: str = mcp_tool["name"]
        full_name = f"{self._prefix}{local_name}"
        description = mcp_tool.get("description") or f"MCP tool {local_name!r}"
        parameters = mcp_tool.get("inputSchema") or {"type": "object", "properties": {}}
        # Normalise to a JSON-Schema object root if the server gave something odd.
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            parameters = {"type": "object", "properties": {}}

        async def _call(**kwargs: Any) -> Any:
            await self.aconnect()
            return await self._transport.call_tool(local_name, kwargs)

        # Carry the MCP tool name on the wrapped function so callers can
        # introspect it (used by tests and observability hooks).
        _call.__mcp_tool_name__ = local_name  # type: ignore[attr-defined]
        _call.__mcp_server_name__ = self.name  # type: ignore[attr-defined]

        return Tool.from_schema(
            name=full_name,
            description=description,
            parameters=parameters,
            func=_call,
        )

    def _filter(self, tools: list[Tool]) -> list[Tool]:
        """Apply allow/deny glob patterns. Patterns match against the FULL
        (namespaced) tool name, so users write ``"github.delete_*"`` not
        ``"delete_*"``."""
        out = tools
        if self._allow:
            out = [t for t in out if any(fnmatch.fnmatchcase(t.name, p) for p in self._allow)]
        if self._deny:
            out = [t for t in out if not any(fnmatch.fnmatchcase(t.name, p) for p in self._deny)]
        return out


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


class MCP:
    """Public factory for :class:`MCPServer` instances."""

    @classmethod
    def stdio(
        cls,
        name: str,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = MCPServer._DEFAULT_CACHE_TTL,
    ) -> MCPServer:
        """Build an MCP server bound to a stdio (subprocess) transport.

        Unlike :meth:`http`, ``allow=`` is **not required** here: stdio
        servers are subprocesses you spawn yourself, so the trust model
        is "you control what runs".  When neither ``allow=`` nor
        ``deny=`` is given a one-shot ``UserWarning`` is emitted as a
        gentle reminder that *every* tool the subprocess advertises
        will be visible to the LLM — pass ``allow=["*"]`` to silence
        the warning once you've audited the surface.
        """
        if allow is None and deny is None:
            warnings.warn(
                f"MCP.stdio({name!r}, command={command!r}) was called without an "
                "explicit allow= or deny= list, so every tool the subprocess "
                "advertises will be exposed to the LLM.  This is fine when you "
                "fully control the subprocess, but pass allow=['tool_a', ...] "
                "to restrict the surface, allow=['*'] to opt in explicitly, or "
                "deny=['dangerous_*'] to block specific patterns.",
                UserWarning,
                stacklevel=2,
            )
        from lazybridge.ext.mcp.transports import StdioTransport

        return MCPServer(
            name,
            transport=StdioTransport(command, args=args, env=env),
            namespace=namespace,
            prefix=prefix,
            allow=allow,
            deny=deny,
            cache_tools_ttl=cache_tools_ttl,
        )

    @classmethod
    def http(
        cls,
        name: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = MCPServer._DEFAULT_CACHE_TTL,
    ) -> MCPServer:
        """Build an MCP server bound to a Streamable HTTP transport.

        ``allow=`` is **required**. Omitting it raises ``ValueError`` because
        a remote server could advertise any number of tools and silently
        exposing them all to the LLM is a security mistake. Pass an explicit
        list of the tools you want to expose::

            MCP.http("github", url, allow=["create_issue", "list_prs"])

        To permit all tools advertised by a server you fully control::

            MCP.http("internal", url, allow=["*"])
        """
        if allow is None:
            raise ValueError(
                f"MCP.http({name!r}, {url!r}) requires an explicit allow= list. "
                f"Every tool the remote server advertises would otherwise be exposed to the LLM. "
                f"Pass allow=['tool_a', 'tool_b'] to restrict the tool surface, "
                f"or allow=['*'] to permit everything and silence this error."
            )
        from lazybridge.ext.mcp.transports import HttpTransport

        return MCPServer(
            name,
            transport=HttpTransport(url, headers=headers),
            namespace=namespace,
            prefix=prefix,
            allow=allow,
            deny=deny,
            cache_tools_ttl=cache_tools_ttl,
        )

    @classmethod
    def from_transport(
        cls,
        name: str,
        transport: _Transport,
        *,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = MCPServer._DEFAULT_CACHE_TTL,
    ) -> MCPServer:
        """Build an MCP server from a custom :class:`_Transport`.

        Useful for tests (in-process fake transport) or for adapters to
        non-standard MCP variants. The transport must implement the
        abstract :class:`_Transport` interface.
        """
        return MCPServer(
            name,
            transport=transport,
            namespace=namespace,
            prefix=prefix,
            allow=allow,
            deny=deny,
            cache_tools_ttl=cache_tools_ttl,
        )
