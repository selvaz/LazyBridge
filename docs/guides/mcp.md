# MCP integration (ext.mcp)

**Use MCP** when a tool catalogue already exists as an MCP server
(`@modelcontextprotocol/server-filesystem`, `mcp-postgres`, GitHub
MCP, your own internal one) and you don't want to write Python
wrappers for each tool.

**Don't use MCP** for tools you'll write yourself — a plain Python
function is shorter, faster, and fully typed.

The integration is at the **tool boundary**.  Once an `MCPServer` is
in `Agent(tools=[...])`, the agent treats its tools exactly like
local functions: parallel calls, structured arguments, cost tracking,
session events.

## Example

```python
from lazybridge import Agent
from lazybridge.ext.mcp import MCP

# 1) Spawn a stdio MCP server (subprocess) and use its tools.
fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
)
agent = Agent("claude-opus-4-7", tools=[fs])
agent("Read README.md and summarise the install steps")

# 2) Mix MCP with custom tools and other agents.
def estimate_cost(plan: str) -> float:
    """Estimate the cost in USD of executing ``plan``."""
    return 0.0

planner = Agent(
    "claude-opus-4-7",
    tools=[fs, estimate_cost],
    name="planner",
)

# 3) Allow / deny lists keep dangerous tools out of reach.
fs_safe = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.list_*", "fs.read_*"],
    deny=["fs.delete_*"],
)

# 4) Refresh the tool list explicitly when an upstream plugin is
#    installed mid-process.
fs.invalidate_tools_cache()

# 5) Explicit lifecycle (rare; the transport otherwise lives until
#    process exit).
async with MCP.stdio("fs", command="...") as fs:
    agent = Agent("claude-opus-4-7", tools=[fs])
    await agent.run("...")
```

## Pitfalls

- Tool-name collisions across servers are real.  Default namespacing
  prevents them; don't disable it casually.
- ``allow`` / ``deny`` patterns match the **namespaced** name; write
  ``"github.delete_*"``, not ``"delete_*"``.
- Lazy connect surfaces transport errors at ``Agent(tools=[server])``
  time, not at first user query.  If the underlying subprocess won't
  start, you'll see the error during agent construction.
- ``cache_tools_ttl=None`` (legacy behaviour) caches forever — fine
  for static MCP servers, dangerous for hot-loaded ones.
- An MCP tool's JSON Schema is published by the server; LazyBridge
  uses it directly via ``Tool.from_schema``.  If the schema is
  malformed, the model will fail the tool call with the new
  ``ToolArgumentParseError`` shape rather than silently coerce.

!!! note "API reference"

    # Status: alpha (lazybridge.ext.mcp).  Install: pip install lazybridge[mcp].
    
    from lazybridge.ext.mcp import MCP, MCPServer
    
    MCP.stdio(
        name: str,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = 60.0,    # tool-list cache lifetime
    ) -> MCPServer
    
    MCP.http(
        name: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = 60.0,
    ) -> MCPServer
    
    MCP.from_transport(
        name: str,
        transport: _Transport,
        *,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        cache_tools_ttl: float | None = 60.0,
    ) -> MCPServer
    
    # MCPServer behaves like a tool provider — drop into Agent(tools=[...])
    # and it expands into one Tool per MCP tool.
    server.invalidate_tools_cache() -> None
    async with server:        # explicit lifecycle: connect + close
        ...

!!! warning "Rules & invariants"

    - An ``MCPServer`` is a *tool provider*; pass it directly to
      ``Agent(tools=[server])``.  ``build_tool_map`` calls
      ``server.as_tools()`` to expand it into one ``Tool`` per MCP tool.
      No separate ``MCPEngine`` / ``MCPProvider`` exists.
    - The transport connects **lazily** on the first ``as_tools()`` call,
      which is normally Agent construction time.  Connection failures
      surface there — fail-fast.
    - Default tool naming: ``"<server-name>.<mcp-tool-name>"``.  Pass
      ``namespace=False`` to keep raw names, or ``prefix="..."`` to
      override.
    - ``allow`` / ``deny`` use shell-style globs (``fnmatch``) against the
      full namespaced name.  ``"github.delete_*"``, not regex.
    - The discovered-tools cache lives ``cache_tools_ttl`` seconds
      (default 60).  An MCP server that hot-loads or unloads tools is
      reflected on the next call past the TTL.  Pass
      ``cache_tools_ttl=None`` to disable expiry, or call
      ``server.invalidate_tools_cache()`` on an out-of-band signal.
    - Closure is **terminal**.  After ``aclose()`` (or exiting an
      ``async with`` block) the server cannot be reconnected — construct
      a new one if you need to.
    - The MCP SDK is an optional dependency.  Importing
      ``lazybridge.ext.mcp`` is cheap; constructing an
      ``MCP.stdio(...)`` / ``MCP.http(...)`` is when the SDK gets imported
      and raises a clean ``ImportError`` if missing.

## See also

- [Tool](tool.md) — local-function tools (the more common path).
- [Recipes/MCP](../recipes/mcp.md) — long-form walkthrough with the
  filesystem and GitHub MCP servers.
