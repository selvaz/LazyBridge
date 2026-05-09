# Recipe: Tool calling

**Tier:** Basic
**Goal:** Give an agent one or more Python functions as tools and get a response that uses them.

No decorators, no JSON schemas — LazyBridge builds the tool schema from type hints and the docstring.

## Minimal case

```python
from lazybridge import Agent

def get_weather(city: str) -> str:
    """Return current temperature and conditions for ``city``."""
    return f"{city}: 22°C, sunny"

# Agent(...) constructs; ("task") invokes; .text() reads the returned Envelope.
agent = Agent("claude-opus-4-7", tools=[get_weather])
print(agent("what's the weather in Rome and Paris?").text())
```

The model may call `get_weather` more than once per turn. All calls in a single turn run
concurrently via `asyncio.gather` — no config needed.

## Multiple tools

```python
from lazybridge import Agent

def search_web(query: str) -> str:
    """Search the web and return top results for ``query``."""
    return "..."

def get_stock_price(ticker: str) -> float:
    """Return the current stock price for ``ticker``."""
    return 142.50

agent = Agent("claude-opus-4-7", tools=[search_web, get_stock_price])
print(agent("what's the latest news on AAPL and its current price?").text())
```

## Inspect which tools were called

`Session` records a `TOOL_CALL` event for every invocation:

```python
from lazybridge import Agent, Session

sess = Session()
agent = Agent("claude-opus-4-7", tools=[get_weather], session=sess)
agent("weather in Rome and Paris?")

for event in sess.events.query(event_type="TOOL_CALL"):
    print(event["payload"])
    # {"name": "get_weather", "input": {"city": "Rome"}}
    # {"name": "get_weather", "input": {"city": "Paris"}}
```

## Agent as tool

Any Agent can be a tool for another Agent — the contract is identical:

```python
from lazybridge import Agent

def search(query: str) -> str:
    """Search the web for ``query``."""
    return "..."

researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")

# editor calls researcher exactly like any other tool
editor = Agent("claude-opus-4-7", tools=[researcher], name="editor")
print(editor("find the top 3 AI papers from April 2026 and summarise them").text())
```

## Pitfalls

- **Type hints aren't optional.** A function without annotations renders an
  empty JSON schema, the model can't fill the arguments, and the tool either
  isn't called or fails with a `ToolArgumentParseError`. Annotate every
  parameter, or pass `mode="llm"` with a `schema_llm=` agent — see
  [Function → Tool](../guides/tool-schema.md).
- **Docstrings are part of the contract.** "Returns the weather" is weaker
  than "Returns the current temperature in Celsius and a one-word condition
  (sunny / cloudy / rainy) for ``city``." Treat the docstring as the prompt
  the model reads when deciding whether to call the tool.
- **Malformed argument JSON now fails loudly.** When a model emits
  un-parseable arguments the engine emits a structured `TOOL_ERROR`
  (`type="ToolArgumentParseError"`) before the tool runs. The tool body is
  not invoked; the model sees the parse error on the next turn and
  self-corrects, instead of failing later with a misleading
  "missing required field" message.
- **Tool name collisions are silently shadowed.** When two tools share a
  name, the framework emits a `UserWarning` once and keeps the last
  registration. Pick stable, unique names — the LLM only ever sees the
  kept one.

## Next

- [Structured output](structured-output.md) — get a typed Pydantic model back instead of plain text
- [Function → Tool schema modes](../guides/tool-schema.md) — when type hints are missing
- [Agent.chain](../guides/chain.md) — linear pipeline passing output between agents
- [MCP integration](mcp.md) — get a whole tool catalogue from an MCP server
