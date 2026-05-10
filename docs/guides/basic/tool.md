# Tool

A `Tool` is anything an `Agent` can call — a Python function, another
agent, an MCP server, an external-tools kit. They all flow through the
same `tools=[...]` list and **you almost never construct one yourself**:
the framework normalises the list inside `Agent.__init__` and registers
each entry under a unique name.

## Signature

```python
from lazybridge import Tool

# Canonical constructor (rarely needed — drop the function in directly).
Tool(
    func,                          # required: the callable
    *,
    name=None,                     # defaults to func.__name__
    description=None,              # defaults to the function's docstring
    mode="signature",              # "signature" | "llm" | "hybrid"
    schema_llm=None,               # engine for mode="llm" / "hybrid"
    strict=False,                  # provider-strict JSON schema validation
    returns_envelope=False,        # set automatically by agent.as_tool()
)

# Pre-built JSON Schema (MCP, OpenAPI, third-party registries).
Tool.from_schema(
    name,                          # required
    description,                   # required
    parameters,                    # JSON Schema dict
    func,                          # the callable to dispatch
    *,
    strict=False,
    returns_envelope=False,
)
```

For the public `tool(...)` factory and `agent.as_tool(...)` method —
both are sugar with non-trivial differences — see
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md).

## Synopsis

LazyBridge accepts six things in `tools=[...]` and normalises them all
to `Tool` instances at construction time:

```python
from lazybridge import Agent, LLMEngine, Tool, tool
from lazybridge.ext.mcp import MCP
from lazybridge.external_tools.read_docs import read_docs_tools

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[
        plain_function,                              # 1. plain Python function
        tool(plain_function, name="custom", strict=True),  # 2. function + overrides via factory
        other_agent,                                 # 3. sub-agent (auto-wrapped)
        other_agent.as_tool(verify=judge),           # 4. sub-agent + judge/retry
        MCP.stdio("fs", command="npx",
                  args=["@modelcontextprotocol/server-filesystem", "."],
                  allow=["fs.read_*", "fs.list_*"]),  # 5. MCPServer (allow= required)
        *read_docs_tools(),                          # 6. external_tools kit (list[Tool])
    ],
)
```

The common case is **path 1**: drop the function in. Type hints +
docstring drive the JSON schema. Reach for `tool(fn, name=..., ...)`
when you need to override the name / description / strictness / mode;
reach for `Tool.from_schema(...)` when you already have a JSON schema
(MCP, OpenAPI, third-party registry). The bare `Tool(...)` constructor
is still public for advanced use cases (e.g. typing annotations,
isinstance checks) but the `tool()` factory is the canonical form
for new code.

## When to construct a Tool explicitly

- **You need a different name than `func.__name__`.** Useful for
  shadowing a third-party function with a clearer LLM-facing name.
- **You need `strict=True`** for provider-strict JSON-schema
  validation (Anthropic / OpenAI strict mode).
- **You need `mode="llm"` or `"hybrid"`** because the function lacks
  type hints or annotations (legacy code, third-party callables,
  `**kwargs`-only signatures).
- **You're shipping a tool kit.** Library authors return
  `list[Tool]` from a factory so callers can splat it into
  `tools=[...]` (e.g. `read_docs_tools()`).

## When NOT to construct a Tool explicitly

- **For ordinary Python functions in your own code.** Just pass the
  function — `Agent(tools=[my_function])` is the canonical form.
- **For an `Agent` you want to use as a sub-agent.** Pass it
  directly: `Agent(tools=[other_agent])`. The agent's `name=` becomes
  the tool name. Use `agent.as_tool("alias")` only to rename or to
  attach `verify=` (see
  [Canonical vs sugar](../../concepts/canonical-vs-sugar.md)).
- **For an MCP server.** Pass the `MCPServer` directly — it's a
  `ToolProvider` and the framework expands it into per-server-tool
  `Tool` instances at construction.

## Example

```python
from lazybridge import Agent, LLMEngine, Tool
from pydantic import BaseModel


# 1) Plain function — type hints + docstring drive the schema.
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)  # noqa: S307  (trusted inputs only)

calc_agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[calculate],
)
result = calc_agent("what is 17 * 23?")
print(result.text())


# 2) Function + explicit configuration — override the name and turn on strict.
calc_tool = tool(
    calculate,
    name="calc",
    description="Evaluate an arithmetic expression and return the numeric result.",
    strict=True,
)
strict_agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[calc_tool],
)


# 3) Pydantic-typed parameters — coerced from the LLM's raw dict to a typed instance.
class SearchInput(BaseModel):
    query: str
    limit: int = 10

def search(input: SearchInput) -> list[str]:
    """Search the web and return the top ``input.limit`` URLs for ``input.query``."""
    return [f"https://example.com/{input.query}/{i}" for i in range(input.limit)]

researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search],
)


# 4) Pre-built schema — Tool.from_schema for MCP / OpenAPI bridges.
schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "units": {"type": "string", "enum": ["c", "f"], "default": "c"},
    },
    "required": ["city"],
}

def weather_dispatch(**kwargs):
    """Forward the validated args to a downstream HTTP call."""
    return f"weather for {kwargs['city']}"

weather_tool = Tool.from_schema(
    name="weather",
    description="Get current weather for a city.",
    parameters=schema,
    func=weather_dispatch,
)
weather_agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[weather_tool],
)
```

## Pitfalls

- **No type hints → empty schema.** A function with bare parameters
  produces an empty JSON schema and the LLM will not know how to
  call it. Always annotate every parameter; defaults are fine.
- **Docstring is the contract.** "Returns the weather" is much
  weaker than "Returns the current temperature in Celsius and a
  one-word condition (sunny / cloudy / rainy) for ``city``." The
  LLM reads the docstring; treat it as the spec.
- **`strict=True` rejects optional / defaulted args** under some
  providers. If a call fails with "unknown parameter", try
  `strict=False`.
- **Name collisions trigger a `UserWarning`.** The second
  registration replaces the first. Pick stable, distinct names —
  especially when mixing your tools with MCP-namespaced ones
  (`fs.read`, `fs.write`, …).
- **`Pydantic BaseModel` parameters are coerced** from the LLM's
  raw dict to a typed instance before your function is called.
  You always receive the model, not the dict — don't write defensive
  `dict(...)` conversions inside the function body.
- **`returns_envelope=True` is set for you** when the framework
  wraps an `Agent` as a tool via `agent.as_tool(...)`. Don't set it
  manually on a plain function — engines that respect the hint will
  try to read `result.metadata.cost_usd` and crash on a non-Envelope
  return value.

## See also

- [Agent](agent.md) — the surface that consumes tools.
- [Native tools](native-tools.md) — provider-hosted alternatives
  passed via `native_tools=[...]` instead of `tools=[...]`.
- [Envelope](envelope.md) — the result type when a tool is an
  `Agent` (`returns_envelope=True`).
- [Everything is a tool](../../concepts/everything-is-a-tool.md) —
  the composition rule that makes all six paths uniform.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  `tool(...)` factory, `agent.as_tool(...)`, and `Tool.from_schema(...)`
  with their canonical equivalents.  Both `tool(...)` and `Tool(...)`
  default to `mode="signature"`; `mode="hybrid"` / `mode="llm"` are
  the explicit opt-ins for LLM-driven schema generation.
