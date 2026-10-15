# Tools and nesting

Read when defining tools, exposing one agent to another, building dynamic
multi-agent graphs, enabling provider-native tools, or connecting MCP servers.
Source of truth: `lazybridge/tools.py`, `lazybridge/pool.py`,
`lazybridge/signals.py`, `lazybridge/core/tool_schema.py`; docs
`docs/guides/basic/tool.md`, `docs/guides/basic/native-tools.md`,
`docs/guides/mid/{as-tool,dynamic-graph,pool-chain}.md`,
`docs/decisions/tool-wrapping.md`.

## Functions are tools

```python
def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    ...


agent = Agent(engine=LLMEngine("claude-opus-5-5"), tools=[get_weather], name="weather")
```

Plain typed callables auto-wrap. Never write the JSON schema by hand when a
Python function exists. Wrap a library's annotated function directly; write a
wrapper only to add an output cap, provenance or an approval boundary, and
have it call the library rather than re-implement it.

## Tool constructors

| Form | Use |
|---|---|
| `Tool(fn, name=, description=, mode="signature", schema_llm=, strict=False, returns_envelope=False, timeout=)` | Explicit tool. `mode`: `"signature"` (default, deterministic), `"hybrid"` or `"llm"` (model-generated schema, needs `schema_llm=`; costs a call). No `"auto"` mode since 0.7.9. |
| `Tool.wrap(x, name=, ...)` | Dispatcher: callable → `Tool`, Agent → `as_tool`, Tool → passthrough/clone. Pin `name=` when Plan/tool-map references must survive refactors. |
| `tool(fn, ...)` | Lowercase alias of `Tool.wrap`, kept for compatibility. |
| `Tool.from_schema(name, description, parameters, func, strict=, returns_envelope=)` | Canonical when a JSON Schema already exists (MCP, OpenAPI, registries). |

Bare `list` / `dict` / `tuple` / `set` annotations compile to the right JSON
type (fixed in 1.3.0). Per-tool deadline: `Tool(timeout=N)` (most specific
wins over `Agent(tool_timeout=)` and `LLMEngine(tool_timeout=)`); see
`agents-and-engines.md` for abandonment semantics.

## Agent as tool

```python
researcher = Agent(engine=LLMEngine("claude-opus-5-5"), name="research", tools=[search])
supervisor = Agent(engine=LLMEngine("claude-opus-5-5"), tools=[researcher], name="supervisor")
```

- The child's `name=` becomes the tool name. Agents passed in `tools=` inherit
  the parent's `Session`.
- `researcher.as_tool("deep_research", verify=judge, max_verify=3)` only to
  rename or to gate every call with a judge loop. `Tool.wrap(researcher,
  name=...)` is the same thing.
- Plan-backed agents and coding-engine agents nest the same way
  (`tools=[pipeline]`, `tools=[codex_reviewer]`).

## Dynamic graph: AgentPool + conclude

```python
from lazybridge import Agent, AgentPool, LLMEngine, conclude

pool = AgentPool(max_depth=25)
alice = Agent(engine=LLMEngine("claude-opus-5-5"), name="alice", tools=[pool.as_tool(), conclude])
bob = Agent(engine=LLMEngine("claude-opus-5-5"), name="bob", tools=[pool.as_tool(), conclude])
pool.register(alice, bob)  # after construction: breaks the cycle
result = alice("...")
```

- `pool.as_tool()` is an ordinary `Tool` named `route(agent_name, task)`;
  `pool.as_tool("ask_team")` renames it so one agent can hold several pools.
- `conclude("answer")` raises `ConcludeSignal` (a `BaseException`) that
  unwinds the whole nested tree back to the top-level run.
- `LLMEngine(max_tool_calls_per_turn=1)` keeps the graph on one path
  (`max_parallel_tools` only bounds concurrency).
- `agent.derive(tools=[pool.as_tool(), conclude])` adds pool tools to an
  existing agent without rebuilding it.

## Provider-native tools

```python
from lazybridge import LLMEngine, NativeTool

engine = LLMEngine("claude-opus-5-5", native_tools=[NativeTool.WEB_SEARCH])
```

`NativeTool` values: `WEB_SEARCH`, `CODE_EXECUTION`, `FILE_SEARCH`,
`COMPUTER_USE`, `IMAGE_GENERATION`, `GOOGLE_SEARCH`, `GOOGLE_MAPS`. They run
server-side at the provider, go on the **engine** (not in `tools=`), and
`CODE_EXECUTION` / `COMPUTER_USE` need `allow_dangerous_native_tools=True`.
An unsupported combination (e.g. `WEB_SEARCH` with DeepSeek) emits a warning and the tool is
dropped; pass `strict_native_tools=True` to the provider to raise
`UnsupportedNativeToolError` instead (`lazybridge.matrix.native_tool_support()` lists support).

## MCP servers and other providers of tools

MCP is **not** in LazyBridge since 0.8 / 0.9 (shims removed). Use LazyTools:

```python
from lazytools.connectors.mcp import MCP  # pip install "lazytoolkit[mcp]"

fs = MCP.stdio(
    "fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    allow=["fs.read_*", "fs.list_*"],  # allow= or deny= is REQUIRED
)
docs = MCP.http("docs", "https://example.com/mcp", allow=["docs.search_*"])
agent = Agent(engine=LLMEngine("claude-opus-5-5"), tools=[fs, docs], name="mcp_user")
```

Tool names are namespaced `"<server>.<tool>"`; allow/deny globs match the full
name. An MCP server is a `ToolProvider`: any object with
`_is_lazy_tool_provider = True` and `as_tools() -> list[Tool]` expands inside
`tools=[...]`. Do not invent a core `MCPEngine` and do not import
`lazybridge.ext.mcp`, `lazybridge.ext.gateway` or `lazybridge.external_tools`.
