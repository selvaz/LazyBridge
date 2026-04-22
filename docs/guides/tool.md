# Tool

You usually never construct a `Tool` yourself. Pass raw callables
straight into `Agent(tools=[my_function])` — LazyBridge wraps each entry
through `wrap_tool` and extracts the JSON schema from type hints +
docstring. The `Tool` class is what comes out of that wrapping, and it is
the uniform representation the engines operate on.

Three things make this one abstraction enough. First, an `Agent`
registered with `tools=[other_agent]` is wrapped via `other_agent.as_tool()`,
which produces a `Tool`. Second, an `Agent` inside an `Agent` inside a
`Tool` is still just a `Tool` to the outer engine. Third, `SupervisorEngine`
and `Plan` accept the same `tools=[...]` surface; you write one tools list
and hand it to any engine.

Reach for an explicit `Tool(...)` call only when you need to override the
name, inject guidance, or switch schema-generation modes.

## Example

Three ways to attach a tool — implicit (plain function), explicit
(`Tool(...)`), and nested (an Agent as a tool).  The LLM-facing schema
is identical for all three; what differs is how much you wanted to
customise before handing it over.

```python
from lazybridge import Tool, Agent

# A plain function with hints + a docstring is a complete tool spec.
# LazyBridge reads `expression: str` for the JSON schema, the return
# annotation to describe the result, and the docstring as the LLM-
# facing description. No JSON to author.
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)  # noqa: S307  (trusted inputs only)

# Implicit: drop the function in tools=. wrap_tool() turns it into
# Tool(calculate) with name=func.__name__ and description=docstring.
Agent("claude-opus-4-7", tools=[calculate])("what is 17 * 23?")

# Explicit: reach for Tool(...) when you need to override the LLM-
# facing tool name, provide a better description, or flip strict=True
# so the provider enforces the schema exactly (Anthropic / OpenAI
# strict mode). Everything else (async/sync dispatch, concurrency,
# error handling) is identical.
calc_tool = Tool(calculate, name="calc", strict=True,
                 description="Evaluate an arithmetic expression.")
Agent("claude-opus-4-7", tools=[calc_tool])("...")

# An Agent is also a Tool — passing researcher to tools= is the same
# as passing researcher.as_tool(). The outer orchestrator sees a tool
# with schema (task: str) -> str; observability of the inner agent's
# run flows into the outer Session automatically.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
orchestrator = Agent("claude-opus-4-7", tools=[researcher])
```

After this, `Tool` rarely shows up in application code directly — most
of the time you pass raw functions or other Agents.  Reach for explicit
`Tool(...)` only when the name, description, strict flag, or schema
mode needs to differ from the function's defaults.

## Async tools

Any `async def` callable works — `Tool.run` awaits it directly and
`Tool.run_sync` drives the coroutine to completion so synchronous
callers (like the SupervisorEngine REPL or non-async test harnesses)
never see a raw coroutine object.

```python
# What this shows: mixing an async DB call and a sync calculator in
# the same tools list. LazyBridge dispatches each tool correctly —
# async functions run on the event loop, sync functions run in an
# executor — and when the LLM emits multiple tool calls in one turn
# the framework fans them out via asyncio.gather.
# Why it matters: you don't have to wrap async code in threadpool
# boilerplate or restructure your API around sync-only tools.

import asyncio
from lazybridge import Agent

async def fetch_user(user_id: int) -> dict:
    """Look up a user record by id."""
    await asyncio.sleep(0)      # stand-in for a real DB / HTTP call
    return {"id": user_id, "name": "Alice"}

def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression."""
    return eval(expression)     # trusted inputs only

Agent("claude-opus-4-7", tools=[fetch_user, calculate])(
    "look up user 7 and compute 3 * (2 + 4)"
)
# Both tools may be emitted in the same turn and run concurrently.
```

## `guidance=` — instructions the LLM follows *when calling*

`description` tells the LLM **what** a tool is (what it does, when to
call it).  `guidance` tells it **how** to call it — invariants on the
arguments, format conventions, anti-patterns.  Both become part of the
tool's LLM-facing contract; keep the description short and the
guidance specific.

```python
# What this shows: stopping a common failure mode (models passing
# natural-language queries when you wanted ISO dates) by spelling out
# the format expectation as guidance.
# Why guidance is separate: description is what the model reads to
# decide whether to call the tool; guidance is what it reads once it
# has decided. Mixing the two produces noisy schemas and
# inconsistent formatting.

from lazybridge import Tool

logs_tool = Tool(
    fetch_logs,
    description="Retrieve application logs for a date range.",
    guidance=(
        "Dates must be ISO-8601 (YYYY-MM-DD). "
        "Pass at most a 7-day window — larger ranges will be rejected. "
        "If the user says 'yesterday' compute the date yourself; do not "
        "pass 'yesterday' as a string."
    ),
)
```

## `returns_envelope=True` — propagating inner Envelope metadata

Most tools return a value (a string, a dict, a BaseModel).  Some —
notably Agents wrapped via `as_tool()` — return an `Envelope` whose
`metadata` carries tokens/cost of an inner LLM call.  Setting
`returns_envelope=True` tells the framework to **propagate** that
inner metadata into the outer Envelope's `nested_*` buckets so
`usage_summary()` sees the full cost tree rather than only the
outermost call.

```python
# What this shows: authoring a tool that calls a sub-agent and
# preserves its token accounting for the outer run's cost summary.
# Why: without returns_envelope=True, the inner agent's cost would be
# invisible at the tool boundary — the outer Envelope would only know
# about the orchestrator's tokens, and session.usage_summary() would
# under-report. as_tool() sets this automatically; you only need it
# when authoring custom tools that return Envelopes directly.

from lazybridge import Agent, Tool, Envelope

inner = Agent("claude-haiku-4-5", name="inner")

async def ask_inner(question: str) -> Envelope:
    """Ask the inner agent; return its full Envelope."""
    return await inner.run(question)

wrapped = Tool(
    ask_inner,
    description="Ask the inner agent a question.",
    returns_envelope=True,     # tells the engine: this returns Envelope,
                               # not a plain value. Merge nested_* metadata.
)

outer = Agent("claude-opus-4-7", tools=[wrapped], name="outer")
```

## Schema modes revisited — when `hybrid` pays off

The three `mode=` values trade off latency, cost, and coverage.  In
most projects you'll never touch this knob — `"signature"` is the
default and works whenever your function is well-typed.  `"hybrid"`
earns its keep in gradually-typed codebases where *some* parameters
have hints but others don't.

```python
# What this shows: wrapping a legacy function with partial type hints.
# Why hybrid: "signature" alone would emit an empty schema for
# ``options``, so the LLM wouldn't know how to populate it. "llm"
# would re-infer every parameter and incur LLM cost even for the
# trivial ones. "hybrid" uses the hints where they exist and only
# calls the schema LLM for ``options``.

from lazybridge import Agent, Tool

tiny = Agent.from_provider("anthropic", tier="cheap", name="schema_bot")

def query_kv(namespace: str, key: str, options=None) -> str:
    """Fetch a value from the namespaced KV store.

    options is a dict controlling TTL and consistency:
      {"ttl": <seconds>, "consistency": "strong"|"eventual"}
    """
    ...

Agent("claude-opus-4-7", tools=[
    Tool(query_kv, mode="hybrid", schema_llm=tiny, strict=True),
])
```

`strict=True` asks the provider (Anthropic / OpenAI strict mode) to
reject any call whose arguments don't match the schema exactly — no
extra fields, no coercion.  Worth combining with `hybrid` when you
want the benefits of LLM-inferred schema for messy args *and* the
guarantee that malformed arguments fail loud instead of silently
trimmed.

## Pitfalls

- A function with no type hints produces an empty JSON schema and the
  LLM will not know how to call it. Always annotate parameters.
- A docstring is part of the contract the LLM reads. "Returns the
  weather" is weaker than "Returns the current temperature in Celsius
  and a one-word condition (sunny / cloudy / rainy) for ``city``."
- ``strict=True`` rejects optional / defaulted args under some providers;
  if a call fails with "unknown parameter", try ``strict=False``.

!!! note "API reference"

    Tool(
        func: Callable,
        *,
        name: str | None = None,
        description: str | None = None,
        guidance: str | None = None,
        mode: Literal["signature", "llm", "hybrid"] = "signature",
        schema_llm: Any | None = None,
        strict: bool = False,
    ) -> Tool
    
    Tool.definition() -> ToolDefinition
    await Tool.run(**kwargs) -> Any
    Tool.run_sync(**kwargs) -> Any   # handles async ``func`` transparently
    
    wrap_tool(obj) -> Tool   # converts functions / Agents / Tools uniformly
    build_tool_map(tools: list) -> dict[str, Tool]

!!! warning "Rules & invariants"

    - Schema generation is automatic from ``func``'s type hints and docstring
      in ``mode="signature"`` (default). Use ``mode="llm"`` to let an LLM
      synthesise the schema, ``mode="hybrid"`` for both.
    - ``name`` defaults to ``func.__name__``. Names are API-facing; pick
      stable ones.
    - ``strict=True`` enables provider-strict JSON-schema validation on tool
      arguments (Anthropic / OpenAI strict mode).
    - ``run`` is async; ``run_sync`` auto-detects coroutine functions and
      drives them to completion so REPL callers (e.g. SupervisorEngine) never
      see a raw coroutine.

## See also

[agent](agent.md), [as_tool](as-tool.md),
decision tree: [parallelism](../decisions/parallelism.md)
