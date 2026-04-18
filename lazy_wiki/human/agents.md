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

### Conversation with memory

Pass a `Memory` object to `chat()` to keep conversation history automatically — no manual list management:

```python
from lazybridge import LazyAgent, Memory

ai  = LazyAgent("anthropic")
mem = Memory()

ai.chat("My name is Marco", memory=mem)
resp = ai.chat("What's my name?", memory=mem)
print(resp.content)   # "Marco"
```

`Memory` accumulates turns automatically and monitors context size. When the conversation gets long, older turns are compressed into a dense summary while recent turns stay raw. The full history is always preserved internally.

```python
print(len(mem))        # 4 — 2 user + 2 assistant messages
print(mem.history)     # full raw history (never truncated)
print(mem.summary)     # compressed block (None if not compressed yet)
mem.clear()            # reset the conversation
```

By default (`strategy="auto"`), compression kicks in when the estimated token count exceeds the budget. You can also force compression or disable it:

```python
Memory()                                    # auto — compresses when needed (default)
Memory(strategy="full")                     # never compress
Memory(strategy="rolling", window_turns=5)  # always use window + compression
Memory(compressor=LazyAgent("openai", model="gpt-4o-mini"))  # LLM compression
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

### Agent-level vs call-level tools

Tools can be attached at **agent construction** or **per-call**. Both are valid — they are merged at runtime:

```python
# Agent-level — tools are always available on every call
specialist = LazyAgent("anthropic", tools=[calculator, search])
specialist.loop("Find data and compute the average")  # both tools available

# Call-level — tools only for this specific call
generalist = LazyAgent("anthropic")
generalist.loop("Calculate 2+2", tools=[calculator])  # only calculator here
generalist.loop("Search for news", tools=[search])    # only search here

# Both combined — agent tools + call tools merged
specialist.loop("Extra task", tools=[extra_tool])  # calculator + search + extra_tool
```

**When to use agent-level tools:** Building specialized agents for pipelines. The agent becomes self-contained — the orchestrator doesn't need to manage its tools:

```python
# Each agent owns its capabilities
researcher = LazyAgent("anthropic", name="researcher", tools=[web_search, arxiv_tool])
analyst = LazyAgent("openai", name="analyst", tools=[calculator, chart_tool])

# Orchestrator just calls agents — doesn't know about their internal tools
orchestrator = LazyAgent("anthropic")
orchestrator.loop("Research and analyze AI trends", tools=[
    researcher.as_tool("research", "Search the web and papers"),
    analyst.as_tool("analyze", "Run calculations"),
])
```

### When to use chat() vs loop()

| Use case | Method |
|----------|--------|
| Single question/answer | `chat()` |
| Need the model to call your functions | `loop()` |
| Complex task that may require multiple tool calls | `loop()` |

### Built-in self-checking with verify=

Pass a `LazyAgent` judge (or any callable) to `verify=` and `loop()` retries automatically if the output is rejected — no manual retry code needed:

```python
judge = LazyAgent(
    "anthropic",
    system=(
        "You are a quality reviewer. "
        "Reply 'approved' if the summary is accurate, self-contained, "
        "and exactly 200 words. Otherwise reply 'rejected: <reason>'."
    ),
)

result = ai.loop(
    "Write a 200-word summary of transformer architecture.",
    tools=[search],
    verify=judge,
    max_verify=2,   # retry up to 2 times (default: 3)
)
print(result.content)
```

The judge sees the output and returns `approved` or `rejected: <reason>`. On rejection, `loop()` re-runs with the judge's feedback appended. On approval (or after `max_verify` attempts), returns the result normally.

Use this for:
- Output length or format constraints
- Factual plausibility checks before passing output to the next agent
- Policy compliance (e.g. "does not contain PII")

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

Receive output token by token as it's generated. Use the dedicated streaming methods for clear return types:

```python
# Preferred — return type is always Iterator[StreamChunk]
for chunk in ai.chat_stream("Write me a haiku about Python."):
    print(chunk.delta, end="", flush=True)
print()  # newline at end
```

Async streaming:

```python
# Preferred — return type is always AsyncIterator[StreamChunk]
async for chunk in await ai.achat_stream("Write a haiku"):
    print(chunk.delta, end="", flush=True)
```

The `chat(stream=True)` form still works but returns a union type (`CompletionResponse | Iterator[StreamChunk]`), which requires runtime type checks. Prefer `chat_stream()` / `achat_stream()` for new code.

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

## Reading the result

After any call, three fields are available:

| Field | Type | Contains |
|---|---|---|
| `agent.result` | `Any` | Best available result: parsed Pydantic object → text content → `None` |
| `agent._last_output` | `str \| None` | Always plain text (used for context injection between agents) |
| `agent._last_response` | `CompletionResponse \| None` | Full response: `.parsed`, `.usage`, `.tool_calls`, `.grounding_sources` |

`agent.result` is the recommended accessor for pipeline code — it automatically surfaces the typed output if the agent had an `output_schema`, otherwise falls back to text:

```python
from pydantic import BaseModel
from lazybridge import LazyAgent

class Summary(BaseModel):
    headline: str
    bullets: list[str]

ai = LazyAgent("anthropic", output_schema=Summary)
ai.chat("Summarise the state of AI in 2024")

# result is a Summary instance (typed), not a string
s = ai.result
print(s.headline)
print(s.bullets)

# For agents without output_schema, result is plain text:
plain = LazyAgent("openai")
plain.chat("Hello")
print(plain.result)   # "Hello! How can I help you today?"
```

For raw token counts and stop reason, use `_last_response`:

```python
resp = ai._last_response
print(resp.usage.input_tokens, resp.usage.output_tokens)
print(resp.stop_reason)
```

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

---

## Human agents

LazyBridge provides two classes for human participation in pipelines:

### HumanAgent — simple approval / review

```python
from lazybridge import HumanAgent, LazyTool, LazyAgent

human = HumanAgent(name="reviewer")

# In a chain — pipeline pauses for human input
pipeline = LazyTool.chain(
    LazyAgent("anthropic", name="researcher"),
    human,
    LazyAgent("openai", name="writer"),
    name="reviewed", description="Research, review, write",
)
result = pipeline.run({"task": "AI safety report"})

# Dialogue mode — multi-turn until "done"
human = HumanAgent(name="reviewer", mode="dialogue")

# As a verify judge
agent.loop("Write code", verify=human, max_verify=3)

# With timeout (auto-approve after 5 min)
human = HumanAgent(name="approver", timeout=300, default="approved")
```

### SupervisorAgent — human with superpowers

```python
from lazybridge import SupervisorAgent, LazySession

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", tools=[search], session=sess)

supervisor = SupervisorAgent(
    name="supervisor",
    tools=[search_tool],           # human can call tools
    agents=[researcher],           # human can retry agents
    session=sess,                  # human can read store
)

# In the REPL:
# > search("AI safety 2026")         — call a tool
# > retry researcher: add 2026 data  — re-run an agent with feedback
# > store findings                   — inspect session store
# > continue                         — pass output forward
```

Both classes work everywhere a LazyAgent works: chains, parallel, as_tool(), verify, LazyContext.from_agent().

### Optional browser UI — `lazybridge.gui.human`

Prefer a browser tab over stdin? Swap the default `input_fn` for the one
provided by the `gui.human` module. Stdlib-only, no extra install.

```python
from lazybridge import SupervisorAgent
from lazybridge.gui.human import web_input_fn

fn = web_input_fn()  # opens a local tab on 127.0.0.1:<ephemeral>
supervisor = SupervisorAgent(name="supervisor", input_fn=fn, ...)
# ... run the pipeline ...
fn.server.close()
```

The page renders each prompt with the previous agent's output, optional
quick-command chips, and a textarea. Ctrl/⌘-Enter submits. Works the same
way for `HumanAgent`. Full API and security notes:
[`lazybridge/gui/human/README.md`](https://github.com/selvaz/LazyBridge/blob/main/lazybridge/gui/human/README.md).
Runnable pipeline:
[`examples/human_gui_demo.py`](https://github.com/selvaz/LazyBridge/blob/main/examples/human_gui_demo.py).
