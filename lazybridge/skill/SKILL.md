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
<https://selvaz.github.io/LazyBridge> are the source of truth.

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
after, with a one-line "use this when …".

### Sugar catalogue (verified against the source)

Several factories are sugar **but not all are pure aliases** — some
build extra structure or return different types. Read the
"Differences" column carefully before substituting.

**Build an Agent with an LLM engine**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent("claude-opus-4-7", **kw)` | `Agent(engine=LLMEngine("claude-opus-4-7"), **kw)` | Pure alias. |
| `Agent.from_model("claude-opus-4-7", **kw)` | `Agent(engine=LLMEngine("claude-opus-4-7"), **kw)` | Pure alias of `Agent("model", …)`. |
| `Agent.from_engine(engine, **kw)` | `Agent(engine=engine, **kw)` | Pure alias. |
| `Agent.from_provider("anthropic", tier="top", **kw)` | `Agent(engine=LLMEngine("top", provider="anthropic"), **kw)` | **Not pure sugar** — uses tier-alias model strings (`super_cheap`/`cheap`/`medium`/`expensive`/`top`) resolved via the provider's tier map. Use when you want freshest-in-tier without pinning a date-stamped name. |

**Build an Agent with a Plan engine**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent.from_plan(*steps, store=…, checkpoint_key=…, resume=…, **kw)` | `Agent(engine=Plan(*steps, store=…, checkpoint_key=…, resume=…), **kw)` | Pure alias; forwards Plan kwargs (`max_iterations`, `store`, `checkpoint_key`, `resume`, `on_concurrent`) to a freshly-built `Plan`. |

**Compose agents — sequential**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent.chain(a, b)` | `Agent(engine=Plan(Step(target=a, name=a.name), Step(target=b, name=b.name)), name="chain")` | **Not pure alias** — builds the `Plan`+`Step` graph for you. Targets are the agents themselves (no `tools=` needed; `Plan` dispatches `Agent` targets via `target.run()` directly). |
| `Agent.from_chain(a, b)` | Identical to `Agent.chain(a, b)` | Pure alias of `.chain`. |

**Compose agents — parallel fan-out**

| Sugar | Canonical | Differences |
|---|---|---|
| `Agent.parallel(*agents, concurrency_limit=…, step_timeout=…)` | (no `Agent`-shaped equivalent) | **Not sugar over `Agent`** — returns `_ParallelAgent`, a sibling class whose `__call__` returns `list[Envelope]`. Closest from-primitives form is hand-written `asyncio.gather(*[a.run(task) for a in agents])`. Use this when you want every branch unconditionally; use `Agent(tools=[a, b, c])` to let the LLM decide; use a `Plan` parallel band (`Step("a", parallel=True)`) when concurrent steps must aggregate via `from_parallel_all`. |
| `Agent.from_parallel(*agents, …)` | Identical to `Agent.parallel(...)` | Pure alias of `.parallel`. Same `_ParallelAgent` return — the asymmetry with the other `from_*` factories is intentional. |

**Build an Agent with a HIL engine**

| Sugar | Canonical | Differences |
|---|---|---|
| `human_agent(timeout=…, ui=…, default=…, **agent_kw)` | `Agent(engine=HumanEngine(timeout=…, ui=…, default=…), **agent_kw)` | Pure alias with kwarg split: HIL-engine kwargs go to `HumanEngine(...)`, `**agent_kw` flows to `Agent(...)`. Lives in `lazybridge.ext.hil` to respect the core/ext import boundary. |
| `supervisor_agent(tools=…, agents=…, store=…, input_fn=…, ainput_fn=…, timeout=…, default=…, **agent_kw)` | `Agent(engine=SupervisorEngine(tools=…, agents=…, store=…, input_fn=…, ainput_fn=…, timeout=…, default=…), **agent_kw)` | Same kwarg-split pattern. |

**Wrap a callable as a Tool**

| Sugar / variant | Canonical | Differences |
|---|---|---|
| `tool(search_web, name="search", description=…)` | `Tool(search_web, name="search", description=…, mode="signature")` | **Not pure alias.** Multi-input dispatcher (callable → Tool, Agent → `as_tool`, Tool → passthrough/clone). **Default `mode="auto"` differs from `Tool(...)`'s `"signature"`**: `auto` tries `signature` then `hybrid`, falling back to `llm` only when `allow_llm_schema=True` and `schema_llm=` is supplied. |
| `Tool.from_schema(name, description, parameters, func, strict=…, returns_envelope=…)` | (no callable-introspection canonical) | **Not sugar over `Tool(callable, …)`** — this is the canonical form when the JSON Schema is already known (MCP, OpenAPI bridges, third-party registries). Bypasses the schema builder. |

**Wrap an Agent as a Tool**

| Sugar | Canonical | Differences |
|---|---|---|
| `tools=[other_agent]` (in another agent) | (this is itself the canonical) | The agent's `name=` becomes the surface tool name. |
| `researcher.as_tool("deep_research")` | A `Tool` whose `func` calls `researcher.run` | **Not pure alias.** Use to **rename** (different surface name than `researcher.name`) or to attach a `verify=` / `max_verify=` judge-and-retry loop — a feature `tools=[researcher]` does **not** expose. |
| `tool(researcher, name="deep_research")` | Identical to `researcher.as_tool("deep_research")` | Pure alias of `as_tool` for agent-like inputs. |

**Call an Agent**

| Form | When |
|---|---|
| `result = agent(task)` (sync) | **Canonical entry point.** `__call__` auto-detects an event loop. |
| `result = await agent.run(task)` | Inside an existing `async def` caller. |
| `async for chunk in agent.stream(task):` | Incremental tokens / events. |

Default model in examples: `claude-opus-4-7`. When a user is learning,
err on the side of the longer canonical form — even if a one-liner
works, the canonical version teaches the shape they will need at every
later rung.

Full reference with worked examples for each row:
<https://selvaz.github.io/LazyBridge/concepts/canonical-vs-sugar/>.

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
  [progressive complexity ladder](https://selvaz.github.io/LazyBridge/concepts/progressive-complexity/)
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

- Full mental model: <https://selvaz.github.io/LazyBridge/concepts/mental-model/>
- Composition rule: <https://selvaz.github.io/LazyBridge/concepts/everything-is-a-tool/>
- The 12-rung complexity ladder: <https://selvaz.github.io/LazyBridge/concepts/progressive-complexity/>
- Per-concept guides: <https://selvaz.github.io/LazyBridge/guides/>
- Runnable recipes: <https://selvaz.github.io/LazyBridge/recipes/>
- API reference: <https://selvaz.github.io/LazyBridge/reference/>
- Errors → fixes: <https://selvaz.github.io/LazyBridge/errors/>
