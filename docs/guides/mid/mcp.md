# MCP

Drop an existing tool catalogue served by an MCP (Model Context
Protocol) server straight into `Agent(tools=[...])`. The framework
expands the server into one `Tool` per advertised tool, namespaces
the names to avoid collisions, and respects allow / deny lists for
sensitive surfaces.

> **Status: alpha.** Install: `pip install lazytoolkit[mcp]`.

## Signature

```python
from lazytools.connectors.mcp import MCP

# Spawn a stdio MCP server as a subprocess.
MCP.stdio(
    name,
    *,
    command,                       # e.g. "npx"
    args=None,                     # list[str]
    env=None,                      # dict[str, str]
    namespace=True,                # prepend "<name>." to each MCP tool name
    prefix=None,                   # custom prefix instead of the server name
    allow=None,                    # REQUIRED — iterable of fnmatch globs; deny-by-default
    deny=None,                     # OR an explicit deny list; one of allow/deny must be set
    cache_tools_ttl=60.0,          # tool-list cache lifetime in seconds; None = never expire
)

# Connect to a remote MCP server over Streamable HTTP.
MCP.http(
    name,
    url,
    *,
    headers=None,
    namespace=True,
    prefix=None,
    allow=None,                    # REQUIRED — http transports are deny-by-default
    deny=None,
    cache_tools_ttl=60.0,
)

# Bring-your-own transport.
MCP.from_transport(name, transport, *, namespace=True, prefix=None,
                   allow=None, deny=None, cache_tools_ttl=60.0)


# An MCPServer is a ToolProvider — pass it directly to Agent(tools=[...]).
server.invalidate_tools_cache()    # force re-discovery on the next as_tools() call

async with server:                 # explicit lifecycle (rare)
    ...
```

The discovered-tools cache lives `cache_tools_ttl` seconds. The MCP
SDK is an optional dependency — `import lazytools.connectors.mcp` is cheap;
constructing an `MCP.stdio(...)` or `MCP.http(...)` is what triggers
the SDK import (and a clean `ImportError` if missing).

## Synopsis

An MCP server is a tool catalogue. The framework consumes it as a
**tool provider**: pass the `MCPServer` directly into
`Agent(tools=[...])`, and at agent construction time the framework
calls `server.as_tools()` to expand it into one `Tool` per MCP tool.
There is no separate `MCPEngine` or `MCPProvider` — the integration
sits at the tool boundary, so once the server is in `tools=[...]`
the agent treats its tools exactly like local Python functions:
parallel calls, structured arguments, cost tracking, session events.

The transport connects **lazily** on the first `as_tools()` call,
which is normally agent construction time. Connection failures
surface there — fail-fast.

By default tool names are namespaced as `"<server>.<mcp-tool>"`
(turn off with `namespace=False`, override with `prefix="..."`).
`allow` / `deny` patterns use fnmatch globs against the namespaced
name, so you write `"github.delete_*"` rather than regex.

## When to use it

- **An MCP server already exists for what you need.**
  `@modelcontextprotocol/server-filesystem`, `@modelcontextprotocol/server-github`,
  `mcp-postgres`, your team's internal MCP — drop it in instead of
  writing per-tool Python wrappers.
- **You want to limit which tools the LLM sees.** Use `allow=` /
  `deny=` to expose a safe slice of a larger catalogue (e.g. read-
  only filesystem operations).
- **You want runtime tool discovery.** The server can hot-load new
  tools; the framework picks them up after the cache TTL or on
  `server.invalidate_tools_cache()`.

## When NOT to use it

- **For tools you'll write yourself.** A plain Python function is
  shorter, faster, fully typed, and has no transport overhead. Use
  MCP only when the tool already exists as an MCP server.
- **For provider-native capabilities.** Web search, code execution,
  file search — those are
  [native tools](../basic/native-tools.md) (`native_tools=[...]`),
  not MCP.

## Example

```python
from lazybridge import Agent, LLMEngine
from lazytools.connectors.mcp import MCP


# 1) Spawn a stdio MCP server (subprocess) and use its tools.
#    ``allow=`` (or ``deny=``) is REQUIRED since 0.7.9 — deny-by-default.
#    Omitting both raises ``ValueError`` at construction so the LLM never
#    silently sees an unaudited filesystem / git / shell tool surface.
fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.list_*", "fs.read_*"],   # read-only slice; deny writes implicitly
)
agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[fs],
)
result = agent("Read README.md and summarise the install steps")
print(result.text())


# 2) Mix MCP with custom Python tools and other agents.
def estimate_cost(plan: str) -> float:
    """Estimate the cost in USD of executing ``plan``."""
    return 0.0


planner = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    tools=[fs, estimate_cost],
    name="planner",
)


# 3) Restrict the surface — read-only filesystem.
fs_safe = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/project"],
    allow=["fs.list_*", "fs.read_*"],
    deny=["fs.delete_*"],
)


# 4) Remote HTTP server — allow= is REQUIRED.
github = MCP.http(
    "github",
    url="https://example.com/github-mcp",
    headers={"Authorization": "Bearer ..."},
    allow=["github.get_*", "github.list_*"],
)


# 5) Force a refresh after upstream tools change.
fs.invalidate_tools_cache()


# 6) Explicit lifecycle (rare; the transport otherwise lives until process exit).
async def use_fs():
    async with MCP.stdio("fs", command="...") as fs:
        agent = Agent(
            engine=LLMEngine("claude-haiku-4-5"),
            tools=[fs],
        )
        await agent.run("…")
```

## Pitfalls

- **Both `MCP.http` and `MCP.stdio` are deny-by-default.** Omitting
  both `allow=` and `deny=` raises `ValueError` at construction
  (since 0.7.9 — the pre-fix `stdio` default warned and proceeded,
  which was unsafe for filesystem / git / shell MCP servers).  Pass
  `allow=["*"]` only after auditing the advertised tool surface,
  or restrict with a glob list.
- **Namespaced names in glob patterns.** `allow=` / `deny=` match
  the *full* namespaced name. Write `"github.delete_*"`, not
  `"delete_*"` — the latter never matches when `namespace=True`.
- **Lazy connect surfaces errors at agent construction.** If the
  underlying subprocess won't start, you'll see the error during
  `Agent(tools=[server])`, not at first user query. Wrap the
  construction site in a try/except if you want graceful
  degradation.
- **`cache_tools_ttl=None` caches forever.** Fine for static
  catalogues, dangerous for hot-loaded ones. Default is 60 s.
- **Closure is terminal.** After `aclose()` (or exiting `async
  with`), the server cannot be reconnected — construct a new one
  if you need to.
- **MCP-published JSON Schema is consumed verbatim** via
  `Tool.from_schema`. A malformed upstream schema makes the
  model's call fail with a `ToolArgumentParseError` shape rather
  than silently coerce.

## See also

- [Tool](../basic/tool.md) — the local-Python-function path; the
  more common choice for tools you author yourself.
- [Native tools](../basic/native-tools.md) — provider-hosted
  capabilities (web search, code exec); not MCP.
- [`examples/llm_assistant/05_mcp_allowlisted.py`](https://github.com/selvaz/LazyBridge/blob/main/examples/llm_assistant/05_mcp_allowlisted.py)
  — runnable allowlist-pattern walkthrough.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  `Tool.from_schema` is what backs every MCP-discovered tool, and
  is its own canonical form (not sugar over `Tool(callable, …)`).
