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

## Next

- [Structured output](structured-output.md) — get a typed Pydantic model back instead of plain text
- [Function → Tool schema modes](../guides/tool-schema.md) — when type hints are missing
- [Agent.chain](../guides/chain.md) — linear pipeline passing output between agents
