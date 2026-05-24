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
    # NB: the legacy mode="auto" graceful-fallback ladder was removed
    # in 0.7.9.  Pass an explicit mode; mode="auto" raises ValueError.
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

For the public `Tool.wrap(...)` factory and `agent.as_tool(...)` method —
both are sugar with non-trivial differences — see
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md).

## `tools=` is **not** `native_tools=`

Two parameters, two runtimes — don't conflate them:

| Parameter | Who executes the tool | What goes in it |
|---|---|---|
| `tools=[...]` | **LazyBridge runtime** (this process) | Python callables, sub-`Agent`s, `MCP*` servers, `Tool` instances |
| `native_tools=[...]` | **The LLM provider's servers** | `NativeTool` enum values (e.g. `NativeTool.WEB_SEARCH`) |

`native_tools` is for server-side tools the provider implements
itself — Anthropic web search, OpenAI image generation, Google
grounding, etc. Dangerous server-side tools
(`NativeTool.CODE_EXECUTION`, `NativeTool.COMPUTER_USE`) additionally
require `allow_dangerous_native_tools=True` on the `Agent` — a
deliberately noisy opt-in, since the executor is no longer in your
process. See [Reference → Providers](../../reference/providers.md)
for the per-provider native-tool support matrix.

The rest of this page documents `tools=` only.

## Synopsis

LazyBridge accepts six things in `tools=[...]` and normalises them all
to `Tool` instances at construction time:

```python
from lazybridge import Agent, LLMEngine, Tool
from lazytools.connectors.mcp import MCP
from lazytools.documents import read_docs_tools

agent = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    tools=[
        plain_function,                              # 1. plain Python function
        Tool.wrap(plain_function, name="custom", strict=True),  # 2. function + overrides via factory
        other_agent,                                 # 3. sub-agent (auto-wrapped)
        other_agent.as_tool(verify=judge),           # 4. sub-agent + judge/retry
        MCP.stdio("fs", command="npx",
                  args=["@modelcontextprotocol/server-filesystem", "."],
                  allow=["fs.read_*", "fs.list_*"]),  # 5. MCPServer (allow= required)
        *read_docs_tools(),                          # 6. lazytools docs kit (list[Tool])
    ],
)
```

The common case is **path 1**: drop the function in. Type hints +
docstring drive the JSON schema. Reach for `Tool.wrap(fn, name=..., ...)`
when you need to override the name / description / strictness / mode;
reach for `Tool.from_schema(...)` when you already have a JSON schema
(MCP, OpenAPI, third-party registry). The bare `Tool(...)` constructor
is still public for advanced use cases (e.g. typing annotations,
isinstance checks) but the `Tool.wrap()` factory is the canonical form
for new code.

## Schema modes — `signature`, `hybrid`, `llm`

Every `Tool` carries a JSON Schema that the LLM uses to call it.
The schema is built once on first use and cached for the lifetime
of the process (an `ArtifactStore` interface lets you persist the
cache across runs if you need to).  The **mode** controls how that
schema is generated.

| Mode | What's inspected | Extra LLM call? | Determinism | Use when |
|---|---|---:|:---:|---|
| `"signature"` (default) | Type hints + docstring of `func` | no | full | The function already has type hints and a useful docstring |
| `"hybrid"` | Signature for types + an LLM for parameter descriptions | one, on first build | high — types are fixed, only descriptions vary | Types exist but docstrings are sparse / outdated / wrong |
| `"llm"` | The function's source code (or stub), interpreted by an LLM | one, on first build | medium — both types and descriptions come from the model | Legacy code with no annotations, `**kwargs`-only signatures, third-party callables you can't modify |

Both `hybrid` and `llm` modes require `schema_llm=<engine>` — usually
a cheap-tier LLM dedicated to schema work — and emit a one-shot
warning if the model returns an empty or under-specified result.

### `signature` — the canonical default

```python
from lazybridge import Tool

def get_weather(city: str, units: str = "c") -> str:
    """Return current weather for ``city``.

    Args:
        city: City name (e.g. "Paris").
        units: "c" for Celsius (default) or "f" for Fahrenheit.
    """
    ...

tool = Tool.wrap(get_weather, name="get_weather")
```

The generated schema (abbreviated):

```json
{
  "type": "object",
  "properties": {
    "city":  {"type": "string", "description": "City name (e.g. \"Paris\")."},
    "units": {"type": "string", "description": "\"c\" for Celsius (default) or \"f\" for Fahrenheit.", "default": "c"}
  },
  "required": ["city"]
}
```

This is deterministic and free.  Reach for the other modes only when
this one can't produce a useful schema.

### `hybrid` — types from signature, descriptions from an LLM

The signature has the truth about parameter types and which are
required.  The LLM only fills in the **descriptions** — the parts
the LLM sees when it's deciding whether and how to call the tool.

```python
from lazybridge import LLMEngine, Tool

schema_llm = LLMEngine("claude-haiku-4-5")  # cheap-tier; runs once per tool

def get_weather(city: str, units: str = "c") -> str:
    # Docstring is missing or unhelpful.
    ...

tool = Tool.wrap(
    get_weather,
    name="get_weather",
    mode="hybrid",
    schema_llm=schema_llm,
)
```

Resulting schema: same `{"type": "string"}` types as the signature
path, but `"description"` fields now come from the LLM analysing
the function source.  Required-vs-optional is **still** decided by
the signature (the `units: str = "c"` default keeps it optional).

When to prefer `hybrid` over `signature`:

- You inherited a function whose docstring lies about the parameters.
- The team is migrating to type-hinted code and isn't ready to
  re-document every helper.
- You want consistent LLM-facing descriptions across a large tool kit
  without hand-writing each one.

When **not** to use it:

- You're shipping a security-sensitive tool.  An LLM-generated
  description could understate the risk (e.g. describe a shell-exec
  tool as "runs a command") and bias the model toward calling it.
  Keep `mode="signature"` and own the description.

### `llm` — full LLM-inferred schema

The signature has no useful information — `def legacy(**kwargs)`,
an undocumented third-party callable, or a function defined inside
a `lambda` you can't annotate.  Hand the source (or a stub) to an
LLM and let it produce the whole tool definition.

```python
from lazybridge import LLMEngine, Tool

schema_llm = LLMEngine("claude-haiku-4-5")

def legacy_lookup(*args, **kwargs):
    """Best-effort lookup against the v1 reporting API.  Accepts
    a record id and an optional output format.  Returns a JSON
    string."""
    ...

tool = Tool.wrap(
    legacy_lookup,
    name="legacy_lookup",
    mode="llm",
    schema_llm=schema_llm,
)
```

The framework keeps two invariants the LLM cannot break:

1. **Required parameters come from the signature first.** If the
   signature says a parameter has no default, it stays required even
   if the LLM forgot it.  A `UserWarning` is logged if the LLM omits
   a signature-required parameter.
2. **Strict mode still applies.**  Passing `strict=True` adds
   `"additionalProperties": false` to the generated schema regardless
   of what the LLM produced.

Calibrate your expectations: this mode is the **slowest** to bootstrap
(one extra LLM round-trip on first use), the **least deterministic**
across runs, and the only mode that can produce a schema the function
can't actually accept (e.g. an extra parameter the LLM invented).
Production use cases should treat it as a one-off migration tool —
once it generates a schema that works, copy the result into
`Tool.from_schema(...)` and pin it.

### Caching

All three modes cache the built schema per `Tool` instance for the
lifetime of the process.  For cross-process persistence — e.g. so a
serverless function doesn't pay the LLM-bootstrap cost on every cold
start — pass an `ArtifactStore` to the tool builder; the
`InMemoryArtifactStore` is the in-process default, and the protocol
is small enough to back with Redis / SQLite / S3 in two methods
(`get` / `put`).  See `lazybridge/core/tool_schema.py` for the
protocol and the in-memory reference implementation.

### Pinning the result

Once an `llm`/`hybrid` mode produces a schema you're happy with,
copy the generated JSON into a `Tool.from_schema(...)` call and
remove the `schema_llm=` dependency.  This is the canonical pattern
for moving from "discover the schema" to "ship the schema":

```python
weather_tool = Tool.from_schema(
    name="get_weather",
    description="Return current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name."},
            "units": {"type": "string", "enum": ["c", "f"], "default": "c"},
        },
        "required": ["city"],
    },
    func=get_weather,
)
```

Now the tool is deterministic, free to load, and reviewable in a PR.

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
    engine=LLMEngine("gpt-5.4-mini"),
    tools=[calculate],
)
result = calc_agent("what is 17 * 23?")
print(result.text())


# 2) Function + explicit configuration — override the name and turn on strict.
calc_tool = Tool.wrap(
    calculate,
    name="calc",
    description="Evaluate an arithmetic expression and return the numeric result.",
    strict=True,
)
strict_agent = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
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
    engine=LLMEngine("gpt-5.4-mini"),
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
    engine=LLMEngine("gpt-5.4-mini"),
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
  `Tool.wrap(...)` factory, `agent.as_tool(...)`, and `Tool.from_schema(...)`
  with their canonical equivalents.  Both `Tool.wrap(...)` and `Tool(...)`
  default to `mode="signature"`; `mode="hybrid"` / `mode="llm"` are
  the explicit opt-ins for LLM-driven schema generation.
