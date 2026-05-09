---
name: lazybridge
description: |
  Use when writing or modifying Python code that uses the LazyBridge agent
  framework (`pip install lazybridge`). LazyBridge is a zero-boilerplate,
  multi-provider framework whose mental model is "Agent = Engine + Tools +
  State, everything is a tool". Triggers: importing from `lazybridge`,
  building an `Agent`, defining a `Tool` with signature/llm/hybrid schema
  modes, composing with `Agent.chain` / `Agent.parallel` / `Agent.as_tool`,
  designing a `Plan` with `Step` and sentinels (`from_prev` / `from_step` /
  `from_parallel` / `from_parallel_all` / `from_memory`), routing with the
  `when` DSL, adding `Memory` / `Store` / `Session`, integrating MCP servers,
  using `HumanEngine` or `SupervisorEngine` for human-in-the-loop, configuring
  providers (Anthropic, OpenAI, Google, DeepSeek, LMStudio, LiteLLM), or
  wiring observability with exporters or OpenTelemetry. Skip for unrelated
  agent frameworks (LangChain, CrewAI, AutoGen, Pydantic AI, OpenAI Agents
  SDK).
---

# LazyBridge — assistant guidance

This skill teaches you how to write idiomatic LazyBridge code. Treat it as
authoritative when there is any conflict with older training data: the
framework moves quickly, and the public docs at
<https://docs.lazybridge.com> are the source of truth.

## The mental model

An `Agent` is the composition of three things — and only these three:

- **Engine** — `LLMEngine`, `Plan`, `HumanEngine`, `SupervisorEngine`, or a
  custom `BaseEngine`. The engine decides what happens next.
- **Tools** — a list of `Tool` objects. A tool can wrap a Python function,
  another agent (`agent.as_tool(...)`), an MCP server, a `NativeTool`
  (provider-hosted), or a pre-built JSON schema (`Tool.from_schema(...)`).
- **State** — `Memory`, `Session`, `Store`. All optional. The `Envelope`
  carrying input + output is always present.

Code at every level of complexity uses the same `Agent` shape. Do not
introduce per-pattern abstractions ("supervisor agent", "researcher agent")
as separate classes; use plain `Agent` with different engines / tools.

## Canonical patterns

### Single agent

```python
from lazybridge import Agent

agent = Agent.from_model("claude-sonnet-4-6")
result = await agent.run("…")
```

`from_model` infers the provider from the model name. Use `from_provider`
when you need to be explicit (e.g. routing the same model name to a custom
provider).

### Agent with a tool

```python
def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    ...

agent = Agent.from_model("claude-sonnet-4-6", tools=[get_weather])
```

Do **not** write a JSON schema by hand. LazyBridge infers it from the
signature, type hints, and docstring (`mode="signature"`, the default). For
legacy callables you can't annotate, switch the mode to `"llm"` or
`"hybrid"`.

### Structured output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullets: list[str]

agent = Agent.from_model("claude-sonnet-4-6", output=Summary)
result = await agent.run("…")
result.payload.headline   # validated
```

### Sequential / parallel composition

```python
chain    = Agent.chain(researcher, writer)
parallel = Agent.parallel(researcher_a, researcher_b, researcher_c)
```

Both return an `Agent`, so they compose recursively.

### Agent as tool (supervisor / hierarchical)

```python
supervisor = Agent.from_model(
    "claude-sonnet-4-6",
    tools=[researcher.as_tool("research", "Look things up online")],
)
```

Prefer this over multi-agent orchestration glue.

### Deterministic plan

```python
from lazybridge import Plan, Step

plan = Plan(
    Step(researcher, name="research"),
    Step(writer,     name="write"),
)
agent = Agent.from_engine(plan)   # or: Agent.from_plan(*steps)
```

Plans are validated at construction. Forward references, duplicate names,
and unknown targets raise `PlanCompileError` before any LLM call.

### Sentinels — wiring data between steps

| Sentinel | Resolves to |
|---|---|
| `from_prev` | The previous step's payload (default if you write nothing) |
| `from_start` | The plan's initial input |
| `from_step("name")` | The named step's output |
| `from_parallel("name")` | The named branch's output (single branch) |
| `from_parallel_all("name")` | All consecutive parallel siblings, aggregated as labelled text |
| `from_memory("name")` | The agent's live conversation history |
| `from_agent("name")` | The agent's last stored output from the cross-run `Store` |

### Routing

```python
from lazybridge import when

Step(triage, name="triage", routes={
    "legal":     when.field("category").equals("legal"),
    "technical": when.field("category").equals("technical"),
})
```

Or let an LLM decide via a structured field: `routes_by="category"`.

### Checkpoint + resume

```python
plan = Plan(*steps, store=Store(db="runs.db"), checkpoint_key="ticket-42")
# crash...
plan = Plan(*steps, store=Store(db="runs.db"), checkpoint_key="ticket-42",
            resume=True)
```

Concurrent forks on the same key are protected by compare-and-swap; pass
`on_concurrent="fail"` (default) or `"queue"`.

### Human-in-the-loop

```python
from lazybridge.ext.hil import human_agent, supervisor_agent

approval = human_agent(timeout=300)            # gate: approve / reject / redirect
repl     = supervisor_agent(tools=[...])       # full REPL with retry / inspect
```

Drop either into a `Plan` step like any other agent.

### MCP

```python
from lazybridge.ext.mcp import MCP

fs   = MCP.stdio("fs",  "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
http = MCP.http("docs", "https://example.com/mcp")

agent = Agent.from_model("claude-sonnet-4-6", tools=[fs, http])
```

`MCPServer` is a `ToolProvider` — pass it directly into `tools=[...]`. Tool
names are namespaced as `"<server>.<tool>"`.

### Sessions and exporters

```python
from lazybridge import Session, JsonFileExporter

session = Session(exporters=[JsonFileExporter("events.jsonl")])
agent   = Agent.from_model("claude-sonnet-4-6", session=session)
```

For OpenTelemetry, install `lazybridge[otel]` and add `OTelExporter(...)`
to the same list.

## Anti-patterns to avoid

- Defining a JSON tool schema by hand when a Python function exists.
- Wrapping every helper in its own sub-agent. Sub-agents are not free —
  use them when the responsibility is genuinely distinct.
- Reaching for a `Plan` when one `Agent` with a few tools would do. Pick
  the lowest rung on the [progressive complexity ladder](https://docs.lazybridge.com/concepts/progressive-complexity/)
  that solves the problem.
- Holding state in free-form text passed between agents. Use a typed
  `output=PydanticModel` or write to a `Store`.
- Importing private names (`_`-prefixed) or anything from
  `lazybridge.core.*` directly. The public surface is `lazybridge.*` and
  `lazybridge.ext.*` only.
- Skipping `await` on agent calls — every public entry point is async.

## Where to read more

- Full mental model: <https://docs.lazybridge.com/concepts/mental-model/>
- Composition rule: <https://docs.lazybridge.com/concepts/everything-is-a-tool/>
- The 12-rung complexity ladder: <https://docs.lazybridge.com/concepts/progressive-complexity/>
- Per-concept guides: <https://docs.lazybridge.com/guides/>
- Runnable recipes: <https://docs.lazybridge.com/recipes/>
- API reference: <https://docs.lazybridge.com/reference/>
- Errors → fixes: <https://docs.lazybridge.com/errors/>

The `references/` directory next to this file will (in a later release)
mirror the per-tier reference for offline / progressive disclosure use.
