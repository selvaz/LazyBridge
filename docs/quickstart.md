# Quickstart

Five minutes to a working agent. Copy-paste the snippets in order.

## The grammar

Every LazyBridge agent has the same shape:

```python
Agent(
    engine=...,    # the brain — decides what happens
    tools=[...],   # the capabilities — what the agent can call
    memory=...,    # the context — conversation history
    session=...,   # observability — event log
)
```

The engine is the only thing that changes. A single LLM call, a
multi-step plan, a human approval gate — all use the same Agent wrapper.

**String shortcut** — `Agent("claude-opus-4-7")` is sugar for
`Agent(engine=LLMEngine("claude-opus-4-7"))`. It's valid everywhere in
this guide. Use the explicit `LLMEngine(...)` form when you need to
configure the engine directly (e.g. `system=`, `max_turns=`,
`thinking=`).

**`as_tool("name")`** — the way one Agent becomes a capability of
another. The name you pass connects the tool to the Plan or the LLM::

    researcher.as_tool("research")   →  tool map key: "research"
    Step("research")                 →  calls the "research" tool
    routes={"research": predicate}   →  routes to the "research" step

Keep reading to see this in practice.

## Install

```bash
pip install lazybridge[anthropic]   # or [openai], [google], [deepseek], [all]
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 1. Hello world (two lines)

```python
from lazybridge import Agent

print(Agent("claude-opus-4-7")("what's the capital of France?").text())
```

## 2. Add a native tool (no code)

Native tools run server-side at the provider — no function to write,
no schema to ship. Just pass an enum.

```python
from lazybridge import Agent, NativeTool

search = Agent("claude-opus-4-7", native_tools=[NativeTool.WEB_SEARCH])
print(search("what happened in AI news this week?").text())
```

Available: `WEB_SEARCH`, `CODE_EXECUTION`, `FILE_SEARCH`,
`COMPUTER_USE`, `GOOGLE_SEARCH`, `GOOGLE_MAPS` (each supported by a
subset of providers — see the [native-tools guide](guides/native-tools.md)).

## 3. Add your own tool

Any Python function with type hints + docstring becomes a tool. No
decorators, no JSON schemas. LazyBridge reads the signature and the
docstring and builds the tool schema automatically; if hints are
missing you can switch to [LLM-inferred schemas](guides/tool-schema.md).

```python
from lazybridge import Agent

def get_weather(city: str) -> str:
    """Return current temperature and conditions for ``city``."""
    return f"{city}: 22°C, sunny"

print(
    Agent("claude-opus-4-7", tools=[get_weather])(
        "what's the weather in Rome and Paris?"
    ).text()
)
```

Behind the scenes, LazyBridge wraps the function as a `Tool`, extracts
the JSON schema from the type hints, passes it to the model, and
executes all returned tool calls concurrently via `asyncio.gather`.

## 4. Structured output

Declare the shape you want and read it off the Envelope. If the model
returns malformed JSON the engine retries up to `max_output_retries`
times with the validation error fed back as context.

```python
from lazybridge import Agent
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    bullets: list[str]

env = Agent("claude-opus-4-7", output=Summary)("summarise LazyBridge in 3 bullets")
print(env.payload.title)
print(env.payload.bullets)
```

## 5. Two agents composed

Build sub-agents first, then give them to an orchestrator via
`as_tool("name")`. The name you pass is how the orchestrator refers to
that capability — in the Plan's Step targets, in LLM tool calls, and in
routing rules.

```python
from lazybridge import Agent, LLMEngine, Plan, Step

def search(query: str) -> str:
    """Search the web for ``query``; return the top 3 hits."""
    return "..."

# Sub-agents — declared first, each with its own engine and tools
researcher = Agent(
    engine=LLMEngine("claude-opus-4-7", system="You are a research specialist."),
    tools=[search],
)
writer = Agent(
    engine=LLMEngine("claude-opus-4-7", system="You are a concise technical writer."),
)

# Dynamic orchestrator — LLM decides when to call which agent
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[
        researcher.as_tool("research"),   # ← name connects tool map to LLM calls
        writer.as_tool("write"),
    ],
)
print(orchestrator("write a summary of AI news April 2026").text())

# Deterministic orchestrator — Plan decides the order
pipeline = Agent(
    engine=Plan(
        Step("research"),          # calls researcher.as_tool("research")
        Step("write"),             # calls writer.as_tool("write")
    ),
    tools=[
        researcher.as_tool("research"),
        writer.as_tool("write"),
    ],
)
print(pipeline("AI news April 2026").text())
```

## 6. Observe what happened

```python
from lazybridge import Agent, Session

sess = Session(console=True)   # stdout tracing, one line per event
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher", session=sess)

print(researcher("summarise this week's AI news").text())

# Cost / tokens / latency breakdown across nested agents.
print(sess.usage_summary())
```

For production observability, use `batched=True` (non-blocking emit)
plus `OTelExporter` (GenAI Semantic Conventions out of the box):

```python
from lazybridge import Agent, Session, JsonFileExporter
from lazybridge.ext.otel import OTelExporter

sess = Session(
    db="events.sqlite",
    batched=True,                     # non-blocking
    on_full="hybrid",                 # default — block on AGENT_*/TOOL_*, drop telemetry
    exporters=[
        JsonFileExporter(path="events.jsonl"),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)
```

## 7. Reliability knobs

The Agent surface includes the production knobs by default — opt
into them per call site:

```python
from lazybridge import Agent

agent = Agent(
    "claude-opus-4-7",
    tools=[search],
    timeout=30.0,                    # total deadline for run()
    max_retries=3,                   # provider transient-error retries
    cache=True,                      # prompt caching where supported
    fallback=Agent("gpt-4o"),         # provider redundancy
)
```

If the model emits a malformed tool-call JSON blob the engine emits a
structured `TOOL_ERROR` with `type="ToolArgumentParseError"` and the
raw arguments — the model gets the real failure on the next turn and
self-corrects, instead of failing later with a misleading
"missing required field" message.

## Next

* Keep going with the [**Getting started guide**](guides/getting-started.md).
* Skim the [**decision trees**](decisions/index.md) if you're unsure
  which tool fits your use case.
* Jump to [**Plan + Step**](guides/plan.md) if you're building a
  production pipeline with typed hand-offs and resume semantics.
* Read [**MCP integration**](recipes/mcp.md) if you want to wire in
  an existing Model Context Protocol server.
