## signature
Tool(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    mode: Literal["signature", "llm", "hybrid"] = "signature",
    schema_llm: Any | None = None,
    strict: bool = False,
    returns_envelope: bool = False,
) -> Tool

Tool.from_schema(name, description, parameters, func, *, strict=False, returns_envelope=False) -> Tool
Tool.definition() -> ToolDefinition
await Tool.run(**kwargs) -> Any
Tool.run_sync(**kwargs) -> Any   # drives async ``func`` to completion

Agent.as_tool(name=None, description=None, *, verify=None, max_verify=3) -> Tool

# Six paths to a Tool — pick by what you have:
Agent(tools=[fn])                                     # plain Python function
Agent(tools=[Tool(fn, name=..., strict=True)])        # function + override
Agent(tools=[other_agent])                            # sub-agent (auto-wrapped)
Agent(tools=[other_agent.as_tool(verify=judge)])      # sub-agent + judge/retry
Agent(tools=[mcp_server])                             # MCPServer (provider expansion)
Agent(tools=read_docs_tools())                        # external_tools kit (list[Tool])

## rules
- Schema generation is automatic from ``func``'s type hints and docstring
  in ``mode="signature"`` (default). Use ``mode="llm"`` to let an LLM
  synthesise the schema, ``mode="hybrid"`` for both.
- ``name`` defaults to ``func.__name__``. Names are API-facing; pick
  stable ones.
- ``strict=True`` enables provider-strict JSON-schema validation on tool
  arguments (Anthropic / OpenAI strict mode).
- ``run`` is async; ``run_sync`` auto-detects coroutine functions and
  drives them to completion so synchronous callers (e.g. ``SupervisorEngine``
  REPL) never see a raw coroutine.
- ``Agent(tools=[...])`` accepts callables, ``Tool`` instances, ``Agent``
  instances, and tool providers (objects with ``_is_lazy_tool_provider =
  True`` + ``as_tools()`` method). Everything is normalised to ``Tool``
  at construction; users never call a wrapper directly.

## narrative
A `Tool` is anything an `Agent` can call: a Python function, another
Agent, or a tool provider (MCP server, external gateway). They all
flow through the same `tools=[...]` list — **you never have to convert
them yourself**. The framework normalises the list inside
`Agent.__init__` and registers each entry under a unique name.

The common case is **drop the function in**. Type hints + docstring
drive the JSON schema. Reach for `Tool(...)` only when you need to
override the name, description, or strictness; reach for
`Tool.from_schema(...)` only when you already have a JSON schema (MCP,
OpenAPI). Reach for `mode="llm"` / `"hybrid"` only when the function
lacks type hints — see [Function → Tool](tool-schema.md).

## example
```python
from lazybridge import Agent, Tool

# 1. Plain function — type hints + docstring drive the schema.
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)  # noqa: S307  (trusted inputs only)

Agent("claude-opus-4-7", tools=[calculate])("what is 17 * 23?")

# 2. Function + explicit configuration (override name, strictness, etc.).
calc_tool = Tool(calculate, name="calc", strict=True,
                 description="Evaluate an arithmetic expression.")
Agent("claude-opus-4-7", tools=[calc_tool])("...")

# 3. An Agent is also a Tool — no ceremony.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
orchestrator = Agent("claude-opus-4-7", tools=[researcher])

# 4. Agent + verifier (judge + retry).
judge = Agent("claude-opus-4-7",
              system="Reply 'approved' or 'rejected: <reason>'.")
orchestrator = Agent("claude-opus-4-7", tools=[
    researcher.as_tool(verify=judge, max_verify=2),
])

# 5. MCP server — drop in, framework expands via as_tools().
from lazybridge.ext.mcp import MCP
fs = MCP.stdio("fs", command="npx",
               args=["@modelcontextprotocol/server-filesystem", "."])
Agent("claude-opus-4-7", tools=[fs])

# 6. external_tools kit — factory returns list[Tool].
from lazybridge.external_tools.read_docs import read_docs_tools
Agent("claude-opus-4-7", tools=read_docs_tools())
```

## pitfalls
- A function with no type hints produces an empty JSON schema and the
  LLM will not know how to call it. Always annotate parameters.
- A docstring is part of the contract the LLM reads. "Returns the
  weather" is weaker than "Returns the current temperature in Celsius
  and a one-word condition (sunny / cloudy / rainy) for ``city``."
- ``strict=True`` rejects optional / defaulted args under some providers;
  if a call fails with "unknown parameter", try ``strict=False``.
- Tool name collisions trigger a ``UserWarning`` — the second
  registration replaces the first. Pick stable, distinct names.
- Pydantic ``BaseModel`` parameters are coerced from the raw LLM dict to
  a model instance before the function is called — you always receive a
  typed object, not a plain dict.

## see-also
- [Function → Tool](tool-schema.md) — schema modes (signature / llm / hybrid).
- [Native tools](native-tools.md) — provider-hosted alternatives (`native_tools=` kwarg).
- [Agent](agent.md) — the surface that consumes tools.
- [Agent.as_tool](as-tool.md) — verifier loop on sub-agent calls.
- [MCP integration](mcp.md) — drop in any MCP server as a tool catalogue.
