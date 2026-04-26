# Model Context Protocol (MCP) — connect to any MCP server

**Use this when** you want an LLM to call tools from a Model Context
Protocol server (`@modelcontextprotocol/server-filesystem`,
`mcp-postgres`, GitHub MCP, your own internal one, …) without writing
Python wrappers for each tool.

LazyBridge integrates MCP **at the tool boundary**: an MCP server is a
*tool collection*, and `Agent(tools=[server])` expands into one
LazyBridge `Tool` per MCP tool. There is no separate `MCPEngine`,
`MCPProvider`, or graph node — `tools=[github]` feels like
`tools=[search, calculator, researcher]`.

Status: **alpha** (see [core-vs-ext policy](../guides/core-vs-ext.md)).
The interface may change between minor releases.

## Install

```bash
pip install lazybridge[mcp]
```

## Quickstart

```python
from lazybridge import Agent
from lazybridge.ext.mcp import MCP

fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
)

agent = Agent("claude-opus-4-7", tools=[fs])
print(agent("Read README.md and summarise the install steps").text())
```

## Multiple servers, mixed transports

```python
from lazybridge import Agent
from lazybridge.ext.mcp import MCP

github = MCP.http(
    "github",
    url="https://example.com/mcp",
    headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
)

db = MCP.stdio("db", command="uvx", args=["mcp-postgres"])

agent = Agent("claude-opus-4-7", tools=[github, db])
agent("List my open PRs and the rows of the `users` table created today.")
```

## Namespacing

By default, an MCP server's tools are exposed with the server `name` as
prefix to avoid collisions: `github.list_repositories`, `fs.read_file`,
`db.query`. Override:

```python
fs = MCP.stdio("fs", command="...", namespace=False)        # raw tool names
gh = MCP.stdio("github", command="...", prefix="gh_")        # custom prefix
```

## Allow / deny lists

Filter the tool catalogue with **shell-style glob patterns** (Python's
`fnmatch.fnmatchcase`). Wildcards: `*` matches any sequence, `?` matches a
single char, `[abc]` matches one of a set. **Not regex** — `fs\.delete_.*`
would not match anything; use `fs.delete_*` instead. Patterns are matched
against the **full namespaced name**.

```python
fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.list_*", "fs.read_*"],
    deny=["fs.delete_*"],
)
```

`allow` and `deny` compose: allow-list filters first, then deny-list
removes from the kept set. If both filter the catalogue down to zero
tools, the agent will simply have no tools from that server — no error.

## Lifecycle

The transport connects **lazily** on the first `as_tools()` call.
In practice this means *Agent construction time*: `Agent(tools=[server])`
calls `build_tool_map()` which calls `server.as_tools()` to discover the
tool catalogue, and that's when the subprocess spawns / HTTP session
opens. Connection failures therefore surface at the `Agent(...)` call —
**not** at the first user query — which is what you want for fail-fast
deployment. For explicit cleanup, use the server as an async context
manager:

```python
async with MCP.stdio("fs", command="...") as fs:
    agent = Agent("claude-opus-4-7", tools=[fs])
    await agent.run("...")
```

Without the context manager, the transport stays open for the process
lifetime; the underlying subprocess is normally cleaned up when the
parent process exits.

## How it works

`MCPServer` carries the marker attribute `_is_lazy_tool_provider = True`.
When you pass it to `Agent(tools=[...])`, `build_tool_map()` calls
`server.as_tools()` to expand it into a list of `Tool` instances built
via `Tool.from_schema(...)` — the MCP server already publishes JSON
Schemas for each tool, so we use them directly instead of re-introspecting
the wrapper function. Each tool's `func` dispatches the call through the
underlying `_Transport`, which talks to the real MCP server over stdio
or Streamable HTTP.

## Testing without a real MCP server

`MCP.from_transport(name, transport)` lets you pass any object that
implements the `_Transport` interface. The unit tests in
`tests/unit/test_mcp.py` use a `FakeTransport` that returns a fixed
catalogue and records `call_tool` invocations — no subprocess, no
network, no SDK dependency.

```python
class FakeTransport(_Transport):
    async def connect(self): ...
    async def list_tools(self): return [{"name": "list_directory", ...}, ...]
    async def call_tool(self, name, arguments): return f"fake: {name}"
    async def close(self): ...

server = MCP.from_transport("fs", FakeTransport())
```

## Pitfalls

- **MCP tool names can collide across servers.** Default namespacing
  prevents that. Don't disable it casually.
- **`allow`/`deny` patterns match the namespaced name.** Write
  `"github.delete_*"`, not `"delete_*"`.
- **Lazy connect surfaces transport errors at `Agent(tools=[server])`
  time**, not at first user query. If the underlying subprocess won't
  start, you'll see the error during agent construction.
- **The MCP SDK is an optional dependency.** Importing
  `lazybridge.ext.mcp` is cheap; constructing a real
  `MCP.stdio(...)` / `MCP.http(...)` is when the SDK gets imported and
  raises a clean `ImportError` if missing.

!!! note "API reference"

    ```python
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
    ) -> MCPServer

    MCP.from_transport(
        name: str,
        transport: _Transport,
        *,
        namespace: bool = True,
        prefix: str | None = None,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
    ) -> MCPServer
    ```

!!! warning "Phase 1 scope"

    This release ships **tools only**. MCP resources (exposable as
    LazyBridge `sources=[...]`) and prompts are planned for later
    phases. The same `MCPServer` will gain `.resource(uri)` and
    `.prompt(name, **vars)` accessors without breaking changes to the
    tool surface.
