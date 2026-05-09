---
name: lazybridge
description: |
  Use when writing or modifying Python code that uses the LazyBridge agent
  framework (`pip install lazybridge`). LazyBridge is a zero-boilerplate,
  multi-provider framework whose mental model is "Agent = Engine + Tools +
  State, everything is a tool". Triggers: importing from `lazybridge`,
  building an `Agent`, defining a `Tool` with signature/llm/hybrid schema
  modes, composing with `Agent.chain` / `Agent.parallel` or by passing one
  agent in another's `tools=[...]`, designing a `Plan` with `Step` and
  sentinels (`from_prev` / `from_step` / `from_parallel` /
  `from_parallel_all` / `from_memory`), routing with the `when` DSL, adding
  `Memory` / `Store` / `Session`, integrating MCP servers, using
  `HumanEngine` or `SupervisorEngine` for human-in-the-loop, configuring
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

- **Engine** — `LLMEngine` (default), `Plan`, `HumanEngine`,
  `SupervisorEngine`, or a custom `BaseEngine`. The engine decides what
  happens next.
- **Tools** — a list of `Tool` objects. A tool can wrap a Python function,
  another agent (just pass it in `tools=[...]`), an MCP server, a
  `NativeTool` (provider-hosted), or a pre-built JSON schema
  (`Tool.from_schema(...)`).
- **State** — `Memory`, `Session`, `Store`. All optional. The `Envelope`
  carrying input + output is always present.

Code at every level of complexity uses the same `Agent` shape. Do not
introduce per-pattern abstractions ("supervisor agent", "researcher agent")
as separate classes; use plain `Agent` with different engines and tools.

## Calling convention — sync is canonical

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
)
result = agent("hello")           # sync — returns Envelope
print(result.text())              # str payload
```

Async and streaming forms exist (`await agent.run(task)`,
`async for chunk in agent.stream(task)`) but are **not** the canonical
introduction. **Do not wrap simple examples in `asyncio.run(main())`**;
LazyBridge agents are synchronous-callable by design, and the example
files in `examples/` follow that convention.

## Style rule — show the canonical form first

When you generate code for the user, the canonical shape is

```python
agent = Agent(
    engine=LLMEngine("model-id"),
    tools=[...],
    name="...",
    ...
)
result = agent(task)
print(result.text())
```

with each constructor argument on its own line and `result = agent(task)`
on a separate line from the print. Lead with this form; treat factories
and string-positional shortcuts as **sugar** and only mention them
after, with a one-line "use this when …":

| Sugar | Canonical equivalent | Use the sugar when |
|---|---|---|
| `Agent("m", ...)` | `Agent(engine=LLMEngine("m"), ...)` | Trivially short snippets where the engine choice is uninteresting (rare in real code; never in tutorials) |
| `Agent.from_model("m", ...)` | `Agent(engine=LLMEngine("m"), ...)` | Never — `Agent("m", ...)` already exists and reads better |
| `Agent.from_engine(e)` | `Agent(engine=e)` | Never — adds no clarity |
| `Agent.from_plan(*steps)` | `Agent(engine=Plan(*steps))` | One-liner where no `tools=[...]` is needed |
| `Agent.from_chain(a, b)` / `Agent.chain(a, b)` | `Agent(engine=Plan(Step("a"), Step("b")), tools=[a, b])` | Purely linear handoff with no router / parallel band / checkpoint |
| `Agent.parallel(a, b)` | (no clean canonical equivalent — it's the canonical form for scripted fan-out → `list[Envelope]`) | You want every branch and the list of results |
| `human_agent(...)` | `Agent(engine=HumanEngine(...))` | One-liner; canonical form when the engine choice should be visible at the call site |
| `supervisor_agent(...)` | `Agent(engine=SupervisorEngine(...))` | Same as above |

Default model in examples: `claude-opus-4-7`. When a user is learning,
err on the side of the longer canonical form — even if a one-liner
works, the canonical version teaches the shape they will need at every
later rung.

## Canonical patterns

### Single agent

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
)
result = agent("hello")
print(result.text())
```

`LLMEngine("claude-opus-4-7")` is what makes this an LLM-driven agent.
Configure the engine in place — `LLMEngine("claude-opus-4-7", system=
"...", max_turns=10, thinking=True, ...)` — instead of reaching for
factory variants. Default model is `claude-opus-4-7`.

### Agent with a tool

```python
from lazybridge import Agent, LLMEngine

def get_weather(city: str) -> str:
    """Return the current weather for ``city``."""
    ...

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[get_weather],
)
result = agent("Weather in Paris?")
print(result.text())
```

Do **not** write a JSON schema by hand. LazyBridge infers it from the
signature, type hints, and docstring (`mode="signature"`, the default).
For legacy callables you can't annotate, switch the mode to `"llm"` or
`"hybrid"` via `Tool(callable, mode="llm")`.

### Structured output

```python
from pydantic import BaseModel
from lazybridge import Agent, LLMEngine

class Summary(BaseModel):
    headline: str
    bullets: list[str]

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    output=Summary,
)
result = agent("Summarise the news")
print(result.payload.headline)    # read .payload, not .text()
```

### Sequential / parallel composition

The canonical sequential form is a `Plan` of named steps — same shape
you'll use for routing, parallel bands, and checkpoints later, so the
mental model stays uniform as the workflow grows:

```python
from lazybridge import Agent, Plan, Step

pipeline = Agent(
    engine=Plan(Step("research"), Step("write")),
    tools=[researcher, writer],
)
```

For a *purely* linear handoff with no other plan features,
`Agent.chain(researcher, writer)` is sugar for exactly the form above —
reach for it when you want a one-liner.

For **scripted** fan-out where you want every branch and a
`list[Envelope]` back, use `Agent.parallel(a, b, c)`. Use a `Plan`
parallel band (`Step("a", parallel=True)`) when concurrent steps need to
aggregate via `from_parallel_all`, and put the candidates in
`tools=[...]` when you want the **LLM** to decide which sub-agent to
dispatch instead of running all of them.

### Agent as tool (supervisor / hierarchical)

```python
from lazybridge import Agent, LLMEngine

researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="research",
    tools=[search],
)
supervisor = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[researcher],
)
```

The researcher's `name=` becomes the tool name the supervisor sees.
Use `researcher.as_tool("alias")` only when you need a surface name
different from the agent's own `name=`. Prefer this over building
bespoke multi-agent orchestration glue.

### Deterministic plan

```python
from lazybridge import Agent, Plan, Step

pipeline = Agent(
    engine=Plan(
        Step("research"),
        Step("write"),
    ),
    tools=[researcher, writer],
)
print(pipeline("Topic: AI agents 2026").text())
```

`Step("name")` references a sub-agent by its `name=`. Plans are validated
at construction — forward references, duplicate names, and unknown
targets raise `PlanCompileError` before any LLM call. `Agent.from_plan(*steps)`
is sugar for the explicit form above.

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

```python
Step("write", task=from_prev, context=from_step("research"))
```

### Routing

```python
from lazybridge import when

Step("triage", routes={
    "legal":     when.field("category").equals("legal"),
    "technical": when.field("category").equals("technical"),
})
```

Or let an LLM decide via a structured field on the step's `output=`:
`Step("triage", routes_by="category")`.

### Checkpoint + resume

```python
from lazybridge import Agent, Plan, Step, Store

store = Store(db="runs.db")

pipeline = Agent(
    engine=Plan(*steps, store=store, checkpoint_key="ticket-42"),
    tools=[...],
)
# crash...
resumed = Agent(
    engine=Plan(*steps, store=store, checkpoint_key="ticket-42", resume=True),
    tools=[...],
)
```

Concurrent forks on the same key are protected by compare-and-swap; pass
`on_concurrent="fail"` (default) or `"queue"`.

### Human-in-the-loop

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine, SupervisorEngine

approval = Agent(engine=HumanEngine(timeout=300), name="approve")
repl     = Agent(engine=SupervisorEngine(tools=[...]), name="repl")
```

`human_agent(timeout=300, name="approve")` and
`supervisor_agent(tools=[...], name="repl")` are sugar for the two lines
above. Both forms return regular `Agent` objects — drop them into a
`Plan`'s `tools=[...]` and reference them from a `Step` like any other
agent.

### MCP

```python
from lazybridge import Agent, LLMEngine
from lazybridge.ext.mcp import MCP

fs   = MCP.stdio("fs",  "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
http = MCP.http("docs", "https://example.com/mcp")

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[fs, http],
)
```

`MCPServer` is a `ToolProvider` — pass it directly into `tools=[...]`.
Tool names are namespaced as `"<server>.<tool>"`.

### Sessions and exporters

```python
from lazybridge import Agent, JsonFileExporter, LLMEngine, Session

session = Session(exporters=[JsonFileExporter("events.jsonl")])
agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    session=session,
)
```

For OpenTelemetry, install `lazybridge[otel]` and add `OTelExporter(...)`
to the same list. Nested agents inherit the session unless they pass
their own.

## Anti-patterns to avoid

- **Wrapping simple examples in `asyncio.run(main())`**. The canonical
  call shape is `agent(task)`. Reach for `await agent.run(task)` only
  inside an existing async caller.
- **Defining a JSON tool schema by hand** when a Python function
  exists. The signature path is the default and covers >95% of real
  callables.
- **Hiding the engine behind sugar.** `Agent.from_model(...)`, the
  string-positional `Agent("claude-opus-4-7")`, and `Agent.from_engine(...)`
  all save a line of code at the cost of hiding which engine the agent
  actually runs. Lead with `Agent(engine=LLMEngine("..."), ...)`,
  especially in tutorials and code reviews.
- **Wrapping every helper in its own sub-agent.** Sub-agents are not
  free — use them when the responsibility is genuinely distinct.
- **Reaching for a `Plan` when one `Agent` with a few tools would do.**
  Pick the lowest rung on the
  [progressive complexity ladder](https://docs.lazybridge.com/concepts/progressive-complexity/)
  that solves the problem.
- **Passing the same agent twice via `agent.as_tool(...)` for both
  positional and tool use** when the agent's own `name=` is already
  unique. `tools=[other_agent]` works; `as_tool("alias")` is only for
  renaming.
- **Holding state in free-form text passed between agents.** Use a typed
  `output=PydanticModel` or write to a `Store`.
- **Importing private names** (`_`-prefixed) or anything from
  `lazybridge.core.*` directly. The public surface is `lazybridge.*` and
  `lazybridge.ext.*` only.

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
