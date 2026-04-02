# LazyAgent — Guide

`LazyAgent` is the only class you interact with directly for LLM calls. Everything else (sessions, tools, context) is optional.

---

## Creating an agent

```python
from lazybridge import LazyAgent

# Minimal — just pick a provider
ai = LazyAgent("anthropic")

# With options
ai = LazyAgent(
    "anthropic",
    name="my_assistant",           # used in logs and when exposing as a tool
    model="claude-sonnet-4-6",     # override provider default
    system="You are a terse assistant. Answer in one sentence.",
    max_retries=3,                 # retry on rate limits / server errors
)
```

### Provider strings

| String | Provider |
|--------|----------|
| `"anthropic"` or `"claude"` | Anthropic Claude |
| `"openai"` or `"gpt"` | OpenAI |
| `"google"` or `"gemini"` | Google Gemini |
| `"deepseek"` | DeepSeek |

---

## chat() — single turn

Send a message, get a response. No tool calls, no loop.

```python
resp = ai.chat("What is 2 + 2?")
print(resp.content)       # "4"
print(resp.usage.input_tokens)
print(resp.stop_reason)   # "end_turn"
```

### Conversazione con memoria

Pass a `Memory` object to `chat()` to keep conversation history automatically — no manual list management:

```python
from lazybridge import LazyAgent, Memory

ai  = LazyAgent("anthropic")
mem = Memory()

ai.chat("My name is Marco", memory=mem)
resp = ai.chat("What's my name?", memory=mem)
print(resp.content)   # "Marco"
```

`Memory` accumulates turns automatically: each call appends the user message and the assistant response to the internal history. The next call sends the full history to the model.

```python
print(len(mem))        # 4 — 2 user + 2 assistant messages
print(mem.history)     # list of {"role": ..., "content": ...} dicts
mem.clear()            # reset the conversation
```

The same `Memory` instance can be shared across multiple agents:

```python
mem = Memory()
agent_a.chat("Remember: the project deadline is Friday", memory=mem)
agent_b.chat("What's the deadline?", memory=mem)   # answers "Friday"
```

For cross-session persistence (remembering conversations between program restarts):

```python
import json
from lazybridge import LazyAgent, Memory, LazySession

sess = LazySession(db="chat.db")
ai  = LazyAgent("anthropic", session=sess)

# Restore previous session
raw = sess.store.read("history")
mem = Memory.from_history(json.loads(raw)) if raw else Memory()

ai.chat("continue from where we left off", memory=mem)

# Save at the end
sess.store.write("history", json.dumps(mem.history))
```

### Passing conversation history manually

```python
from lazybridge.core.types import Message, Role

history = [
    Message(role=Role.USER,      content="My name is Alice."),
    Message(role=Role.ASSISTANT, content="Hello Alice!"),
]
resp = ai.chat(history + [Message(role=Role.USER, content="What's my name?")])
print(resp.content)  # "Alice"
```

Or as plain dicts:

```python
history = [
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user",      "content": "What's my name?"},
]
resp = ai.chat(history)
```

---

## loop() — agentic tool calls

`loop()` repeatedly calls the LLM until it stops requesting tools or `max_steps` is reached.

```python
from lazybridge import LazyTool

def search_web(query: str) -> str:
    """Search the web and return results."""
    return f"Results for '{query}': [article1, article2, ...]"

search = LazyTool.from_function(search_web)

result = ai.loop(
    "Find the latest news about fusion energy and summarise it.",
    tools=[search],
    max_steps=8,         # hard cap (default: 8)
)
print(result.content)
```

### When to use chat() vs loop()

| Use case | Method |
|----------|--------|
| Single question/answer | `chat()` |
| Need the model to call your functions | `loop()` |
| Complex task that may require multiple tool calls | `loop()` |

---

## text() and json() shortcuts

```python
# Returns str directly (no CompletionResponse wrapper)
answer = ai.text("What is the speed of light?")

# Returns a typed object (Pydantic model or dict)
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullets: list[str]

summary = ai.json("Summarise AI in 2024", Summary)
print(summary.headline)
print(summary.bullets)
```

---

## Streaming

Receive output token by token as it's generated.

```python
for chunk in ai.chat("Write me a haiku about Python.", stream=True):
    print(chunk.delta, end="", flush=True)
print()  # newline at end
```

Async streaming:

```python
async for chunk in await ai.achat("Write a haiku", stream=True):
    print(chunk.delta, end="", flush=True)
```

---

## System prompt

```python
ai = LazyAgent(
    "anthropic",
    system="You are a Python expert. Only use Python code in your answers.",
)

# Add a per-call addition (appended to the agent-level system)
resp = ai.chat("Show me a list", system="Use bullet points.")
```

---

## Native provider tools

Activate built-in server-side tools (web search, code execution, etc.) with `native_tools`:

```python
from lazybridge.core.types import NativeTool

resp = ai.chat(
    "What happened in AI this week?",
    native_tools=[NativeTool.WEB_SEARCH],
)
print(resp.content)

# Citations are available in grounding_sources
for src in resp.grounding_sources:
    print(src.url, src.title)
```

Works in `loop()` too, and can be combined with your own `LazyTool` functions.

See [Tools guide](tools.md#native-provider-tools) for the full provider support table and grounding details.

---

## Thinking (extended reasoning)

Available on models that support chain-of-thought:

```python
resp = ai.chat("What is 17 × 23?", thinking=True)
print(resp.thinking)   # internal reasoning
print(resp.content)    # final answer
```

For fine-grained control:

```python
from lazybridge.core.types import ThinkingConfig

resp = ai.chat(
    "Design a distributed caching system",
    thinking=ThinkingConfig(enabled=True, effort="high", budget_tokens=8000),
)
```

---

## Controlling tool choice

By default the model decides when to call tools (`"auto"`). You can override this:

```python
# Force the model to call at least one tool
result = ai.loop("Summarise today's news", tools=[search], tool_choice="required")

# Prevent any tool calls
resp = ai.chat("Just answer from memory", tools=[search], tool_choice="none")

# Force a specific tool
result = ai.loop("Find news", tools=[search, calc], tool_choice="search_web")
```

---

## as_tool() — expose this agent to another

Wrap this agent as a callable tool for an orchestrator:

```python
researcher = LazyAgent("anthropic", name="researcher", description="Researches any topic online")
research_tool = researcher.as_tool()

# Optional overrides
research_tool = researcher.as_tool(
    name="web_researcher",
    description="Deep-dives into a topic using multiple web searches",
    guidance="Use this whenever fresh external information is needed.",
)

orchestrator = LazyAgent("anthropic")
result = orchestrator.loop("Write a report on climate tech startups", tools=[research_tool])
```

The orchestrator passes a `task` string; the researcher's `loop()` receives it.

---

## Async

Every method has an async counterpart:

```python
import asyncio

async def main():
    resp   = await ai.achat("Hello")
    result = await ai.aloop("Find news", tools=[...])
    text   = await ai.atext("Hello")
    data   = await ai.ajson("...", MyModel)

asyncio.run(main())
```
