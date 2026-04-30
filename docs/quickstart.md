# Quickstart

Five minutes to a working agent. Copy-paste the snippets in order.

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

## 5. Two agents, chained

Pass one agent as a tool to another. **No ceremony — tool is tool.**

```python
from lazybridge import Agent

def search(query: str) -> str:
    """Search the web for ``query``; return the top 3 hits."""
    return "..."

researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
writer     = Agent("claude-opus-4-7", tools=[researcher], name="writer")

print(writer("write a one-paragraph summary of AI news April 2026").text())
```

Or use the `chain` sugar for a pre-scripted linear pipeline:

```python
print(Agent.chain(researcher, writer)("AI news April 2026").text())
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
        JsonFileExporter("events.jsonl"),
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
    fallback=Agent("gpt-5"),         # provider redundancy
)
```

If the model emits a malformed tool-call JSON blob the engine emits a
structured `TOOL_ERROR` with `type="ToolArgumentParseError"` and the
raw arguments — the model gets the real failure on the next turn and
self-corrects, instead of failing later with a misleading
"missing required field" message.

## Next

* Keep going with the [**Basic tier guide**](tiers/basic.md).
* Skim the [**decision trees**](decisions/index.md) if you're unsure
  which tool fits your use case.
* Jump to [**Plan + Step**](guides/plan.md) if you're building a
  production pipeline with typed hand-offs and resume semantics.
* Read [**MCP integration**](recipes/mcp.md) if you want to wire in
  an existing Model Context Protocol server.
