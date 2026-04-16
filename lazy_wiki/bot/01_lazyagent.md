# LazyAgent — Complete Reference

## 1. Overview

`LazyAgent` is the single entry point for all LLM interaction. The only difference between a stateless and a stateful agent is the `session` parameter. With no session the agent is self-contained; with a session it participates in a shared store, event log, and graph topology.

```python
from lazybridge import LazyAgent

# Stateless
ai = LazyAgent("anthropic")
print(ai.chat("hello").content)

# Stateful
from lazybridge import LazySession
sess = LazySession()
ai = LazyAgent("anthropic", session=sess)
```

---

## 2. Constructor — every parameter

```python
LazyAgent(
    provider: str | BaseProvider,
    *,
    name: str | None = None,
    description: str | None = None,
    model: str | None = None,
    system: str | None = None,
    context: LazyContext | Callable[[], str] | None = None,
    tools: list[LazyTool | ToolDefinition | dict] | None = None,
    native_tools: list[NativeTool | str] | None = None,
    output_schema: type | dict | None = None,
    session: LazySession | None = None,
    verbose: bool = False,
    max_retries: int = 0,
    api_key: str | None = None,
    **kwargs,
)
```

### `provider: str | BaseProvider`

Accepted strings (case-insensitive):

| String | Provider | Env var read |
|---|---|---|
| `"anthropic"` / `"claude"` | Anthropic | `ANTHROPIC_API_KEY` |
| `"openai"` / `"gpt"` | OpenAI | `OPENAI_API_KEY` |
| `"google"` / `"gemini"` | Google | `GOOGLE_API_KEY` |
| `"deepseek"` | DeepSeek | `DEEPSEEK_API_KEY` |

A `BaseProvider` instance may be passed directly to bypass string resolution.

### `name: str | None = None`
Human-readable label. Defaults to the first 8 characters of the agent's UUID (`self.id[:8]`). Used in:
- `sess.graph` node labels
- event log `agent_name` field
- tool name when this agent is exposed via `as_tool()`

### `description: str | None = None`
Default description when the agent is exposed via `as_tool()`. The orchestrator sees this as the tool description.

### `model: str | None = None`
Override the provider's default model. If `None`, the provider selects its own default.

```python
ai = LazyAgent("anthropic", model="claude-opus-4-5")
```

### `system: str | None = None`
Base system prompt. Combined with `context` and tool guidance before every request (see section 3).

### `context: LazyContext | Callable[[], str] | None = None`
Agent-level context. Evaluated and injected into the system prompt at every call. Can be overridden per-call via `chat(context=...)`. A plain callable `() -> str` is accepted and called at build time.

### `tools: list[LazyTool | ToolDefinition | dict] | None = None`
Agent-level tools. Merged with per-call tools in `loop()` / `chat()`. These are the tools used when this agent is invoked as a delegated tool (via `as_tool()`).

### `native_tools: list[NativeTool | str] | None = None`
Provider-managed tools (e.g. `NativeTool.WEB_SEARCH`, `NativeTool.CODE_EXECUTION`). No Python callable needed — executed server-side by the provider. Merged with per-call `native_tools`. Accepts both `NativeTool` enum values and plain strings.

### `output_schema: type | dict | None = None`
Agent-level structured output schema. If set, every `chat()` / `loop()` call returns `resp.parsed` as a validated Pydantic instance (or dict). Can be overridden per-call. When this agent is used in a `mode="chain"` pipeline, `json()` is called automatically instead of `chat()`.

### `session: LazySession | None = None`
If set:
- activates event tracking scoped to this agent
- gives the agent access to `sess.store` (shared blackboard)
- registers the agent in `sess.graph` on construction

### `verbose: bool = False`
Prints all tracked events to stdout in real-time as the agent runs — no DB, no extra setup.

- **Standalone agent** (no `session`): creates a private `EventLog` with `console=True`. Events are printed but not stored.
- **Session agent** (`session=sess`): enables `console=True` on the shared `EventLog`. All agents in that session will print events (including other agents already registered).

```python
# Standalone — see what a single agent is doing
ai = LazyAgent("anthropic", verbose=True)
ai.loop("research AI trends")

# Session — verbose on any agent activates console for the whole session
sess = LazySession(tracking="basic")
researcher = LazyAgent("anthropic", name="researcher", session=sess, verbose=True)
analyst    = LazyAgent("openai",    name="analyst",    session=sess)
# Both agents will print events
```

Alternatively, enable console on the session directly:

```python
sess = LazySession(tracking="basic", console=True)
```

### `max_retries: int = 0`
Retry on HTTP 429 / 5xx. Uses exponential backoff with ±10% jitter.

### `api_key: str | None = None`
Override the provider's environment variable. Use when running multiple providers simultaneously or loading keys from a custom source.

```python
ai = LazyAgent("anthropic", api_key="sk-ant-...")
```

### `**kwargs`
Forwarded to the provider constructor (e.g. custom base URLs, timeouts).

---

## 3. System prompt assembly

Before each LLM request the final system string is assembled in this order:

```
1. agent.system          (base, set at construction)
2. extra_system          (call-level system= kwarg, appended with \n\n)
3. context text          (call-level context overrides agent-level context for this call)
4. tool guidance         (guidance strings from all active LazyTools, appended last)
```

Implementation:

```python
# Step 1+2: _build_system() joins agent.system and extra_system
# Call-level context replaces (does not append to) agent-level context
effective_ctx = call_context or self.context
ctx_text = effective_ctx() if callable(effective_ctx) else effective_ctx.build()

# Tool guidance: one block per LazyTool that has guidance set
# Format: "[tool_name]\n<guidance text>" — joined with \n\n

# Final assembly — all parts joined with "\n\n", None if empty
parts = []
if base_system:    parts.append(base_system)   # agent.system + extra_system
if ctx_text:       parts.append(ctx_text)       # evaluated context string
if tool_guidance:  parts.append(tool_guidance)  # per-tool guidance blocks
system = "\n\n".join(parts) or None
```

---

## 4. `chat()` — single-turn

### Signature

```python
def chat(
    self,
    messages: str | list,
    *,
    memory: Memory | None = None,
    system: str | None = None,
    tools: list | None = None,
    native_tools: list[NativeTool | str] | None = None,
    output_schema: type | dict | None = None,
    thinking: bool | ThinkingConfig = False,
    skills: list[str] | None = None,
    stream: bool = False,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    tool_choice: str | None = None,        # "auto"|"none"|"required"|"<tool_name>"
    context: LazyContext | Callable[[], str] | None = None,
    **kwargs,
) -> CompletionResponse | Iterator[StreamChunk]:
```

`messages` accepts a plain string or a list of `Message` objects / dicts with `role`/`content` keys.

`memory` accepts a `Memory` instance for automatic history accumulation (requires `messages` to be `str`). Each call prepends the stored history and appends the new turn on success.

`context` overrides the agent-level context for this call only. `**kwargs` are forwarded to the provider.

### Memory — stateful conversation

```python
from lazybridge import Memory

mem = Memory()
ai.chat("my name is Marco", memory=mem)
resp = ai.chat("what is my name?", memory=mem)
# resp.content contains "Marco"

len(mem)          # 4 (2 user + 2 assistant)
mem.history       # list[dict] — read-only copy
mem.clear()       # reset

# Shared across agents
agent_a.chat("remember: deadline is Friday", memory=mem)
agent_b.chat("what's the deadline?", memory=mem)
```

`Memory._build_input(msg)` — returns history + new user msg without mutating.
`Memory._record(user, assistant)` — appends a completed turn (called internally by chat() on success).
`stream=True` + `memory` — history NOT updated (content unavailable in streaming chunks).

### Examples

```python
from lazybridge import LazyAgent, LazyContext
from pydantic import BaseModel

ai = LazyAgent("anthropic")

# Basic
resp = ai.chat("What is the capital of France?")
print(resp.content)                    # "Paris"

# Call-level context override
ctx = LazyContext.from_text("Answer in Italian")
resp = ai.chat("What is the capital of France?", context=ctx)
print(resp.content)                    # "Parigi"

# Call-level system addition (appended to agent.system)
resp = ai.chat("What is the capital of France?", system="Reply in one word only.")

# Streaming (preferred — unambiguous return type)
for chunk in ai.chat_stream("Tell me a story"):
    print(chunk.delta, end="", flush=True)

# Streaming (legacy — returns union type, still works)
for chunk in ai.chat("Tell me a story", stream=True):
    print(chunk.delta, end="", flush=True)

# Structured output
class City(BaseModel):
    name: str
    country: str

resp = ai.chat("Give me a city", output_schema=City)
city: City = resp.parsed              # validated Pydantic instance

# Extended thinking
resp = ai.chat("Solve this puzzle: ...", thinking=True)
print(resp.thinking)                  # reasoning trace
print(resp.content)                   # final answer

# Native provider tools (server-side, no Python callable required)
from lazybridge.core.types import NativeTool
resp = ai.chat("What happened in AI this week?", native_tools=[NativeTool.WEB_SEARCH])
```

### `CompletionResponse` fields

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Text output |
| `thinking` | `str \| None` | Reasoning trace (if `thinking=True`) |
| `tool_calls` | `list[ToolCall]` | Tool calls requested by the model |
| `parsed` | `Any` | Pydantic instance or dict when `output_schema` is set |
| `validated` | `bool \| None` | `True` on schema success, `False` on failure, `None` if not requested |
| `validation_error` | `str \| None` | Error message when validation failed |
| `usage` | `UsageStats` | `.input_tokens`, `.output_tokens`, `.thinking_tokens`, `.cost_usd` |
| `stop_reason` | `str` | `"end_turn"`, `"tool_use"`, `"max_tokens"`, etc. |
| `grounding_sources` | `list[GroundingSource]` | Web citations (search grounding) |
| `web_search_queries` | `list[str]` | Queries issued by the grounding tool |
| `raw` | `Any` | Original provider response object |

`resp.raise_if_failed()` raises `StructuredOutputParseError` or `StructuredOutputValidationError` when structured output validation failed; no-op otherwise.

---

## 5. `loop()` — tool loop

### Lifecycle

```
Step 1: call chat() with current conversation
Step 2: if response has tool_calls → execute each tool
Step 3: append assistant turn + tool results to conversation
Step 4: repeat from Step 1
Terminates: when model produces no tool_calls OR max_steps reached
Returns: final CompletionResponse
```

### Signature

```python
def loop(
    self,
    messages: str | list,
    *,
    tools: list | None = None,
    native_tools: list[NativeTool | str] | None = None,
    max_steps: int = 8,
    tool_runner: Callable[[str, dict], Any] | None = None,
    on_event: Callable[[str, Any], None] | None = None,
    verify: LazyAgent | Callable[[str, str], str] | None = None,
    max_verify: int = 3,
    **chat_kwargs,
) -> CompletionResponse:
```

Call-level `tools` are merged with agent-level `self.tools`. `**chat_kwargs` are forwarded to every internal `chat()` call (e.g. `system`, `context`, `temperature`). `max_steps` must be >= 1 or `ValueError` is raised.

**`verify` / `max_verify`:** optional quality gate. Pass a `LazyAgent` (calls `.text()`) or a callable `(question, answer) → verdict`. Return a string starting with `"approved"` (case-insensitive) to accept; anything else triggers a retry with the feedback appended. If all `max_verify` attempts are rejected, `loop()` emits a `UserWarning` and returns the last result unchanged — no exception is raised.

### `on_event` callback

Events fired during `loop()` / `aloop()`:

| Event | Payload | When |
|---|---|---|
| `"step"` | `{"step": int, "response": CompletionResponse}` | After every LLM call |
| `"tool_call"` | `ToolCall` | Before each tool is executed |
| `"tool_result"` | `{"call": ToolCall, "result": Any}` | After each tool returns |
| `"done"` | `CompletionResponse` | When the worker loop finishes a full attempt |
| `"verify_rejected"` | `{"attempt": int, "verdict": str}` | When the judge rejects — includes full feedback |

```python
from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

weather_tool = LazyTool.from_function(get_weather)
ai = LazyAgent("anthropic")

def watch(event: str, payload):
    if event == "tool_call":
        print(f"Calling: {payload.name}({payload.arguments})")
    elif event == "tool_result":
        print(f"Result: {payload['result']}")
    elif event == "verify_rejected":
        print(f"Attempt {payload['attempt']} rejected: {payload['verdict']}")
    elif event == "done":
        print(f"Final: {payload.content}")

result = ai.loop("What is the weather in Rome?", tools=[weather_tool], on_event=watch)
print(result.content)
```

The returned `CompletionResponse` also exposes all judge feedback via `.verify_log: list[str]` — one entry per rejection, in order. Empty if `verify=None` or approved on the first attempt.

**Exhaustion warning:** if all `max_verify` attempts are rejected, `loop()` (and `aloop()`) emit a `UserWarning`:
```
UserWarning: loop() verify exhausted after N attempt(s) without approval.
Returning last result unchanged. Increase max_verify= or review your verify function.
```
Catch or filter it with `warnings.catch_warnings()` if you handle exhaustion programmatically.

### `tool_runner` for tools without a callable

Use `tool_runner` when tools have no Python callable — e.g. raw `ToolDefinition` items or externally handled tools. The runner receives `(name: str, args: dict)` and returns any value.

```python
from lazybridge.core.types import ToolDefinition

lookup_def = ToolDefinition(
    name="lookup_price",
    description="Look up the price of a product",
    parameters={
        "type": "object",
        "properties": {"product": {"type": "string"}},
        "required": ["product"],
    },
)

def my_runner(name: str, args: dict):
    if name == "lookup_price":
        return f"${42 + len(args['product'])}.00"
    raise RuntimeError(f"Unknown tool: {name}")

ai = LazyAgent("anthropic")
result = ai.loop("What does 'widget' cost?", tools=[lookup_def], tool_runner=my_runner)
print(result.content)
```

If a tool name is in neither the LazyTool registry nor `tool_runner`, a `RuntimeError` is raised.

---

## 6. `text()` / `json()` shortcuts

`text()` and `json()` call `chat()` and unwrap the result directly.

```python
from lazybridge import LazyAgent
from pydantic import BaseModel

ai = LazyAgent("anthropic")

# text() → str
answer = ai.text("What is 2 + 2?")
print(answer)           # "4"

# json() → typed object (calls chat() with output_schema= internally)
class CityList(BaseModel):
    cities: list[str]

data = ai.json("List 3 European capitals", schema=CityList)
print(data.cities)      # ["Paris", "Berlin", "Madrid"]
```

Both accept the same `**kwargs` as `chat()`.

### JSON enforcement

`json()` and `ajson()` use a belt-and-suspenders approach to guarantee clean JSON output:

1. **Native structured output API** — passes `output_schema` to the provider (Anthropic constrained sampling, OpenAI strict mode, etc.)
2. **System prompt injection** — appends the following suffix to the system prompt on every call:
   > "Respond with a single valid JSON object matching the required schema. No preamble, no explanation, no markdown — JSON only."

This dual enforcement prevents long or complex responses from drifting into markdown or prose wrapping, which some models produce even with native structured output enabled. The suffix is applied automatically — no action needed from the caller.

When an agent is declared with `output_schema=` and participates in a `mode="chain"` pipeline, the chain calls `json()` / `ajson()` automatically at each step.

---

## 7. `as_tool()` — expose agent as a tool

Wraps this agent as a `LazyTool` so an orchestrator can call it. The tool schema is always `{"task": str}`. The orchestrator passes a task string; the agent's `loop()` or `chat()` receives it.

### Signature

```python
def as_tool(
    self,
    name: str | None = None,
    description: str | None = None,
    *,
    guidance: str | None = None,
    output_schema: type | dict | None = None,
    native_tools: list | None = None,
    system_prompt: str | None = None,
    strict: bool = False,
) -> LazyTool:
```

`name` defaults to `self.name`; `description` defaults to `self.description`. All parameters are forwarded to `LazyTool.from_agent()`.

### Example

```python
from lazybridge import LazyAgent

researcher = LazyAgent("anthropic", name="researcher", description="Searches the web")

# Minimal — uses agent.name and agent.description
researcher_tool = researcher.as_tool()

# With overrides
researcher_tool = researcher.as_tool(
    name="web_researcher",
    description="Search and summarize web content on any topic",
    guidance="Always cite your sources in the response.",
)

orchestrator = LazyAgent("anthropic")
result = orchestrator.loop("Prepare a report on AI trends", tools=[researcher_tool])
print(result.content)
```

When the orchestrator calls the tool, `task` is forwarded to `researcher`. If `researcher` has bound tools (`self.tools`) or `native_tools` is set, `researcher.loop(task, ...)` is called; otherwise `researcher.chat(task)` is called. If `output_schema` is set, `researcher.chat(task, output_schema=...)` is called and `resp.parsed` is returned.

---

## 8. Agent result state — `result`, `_last_output`, `_last_response`

LazyBridge keeps three complementary fields after every call. Understanding their distinct roles prevents confusion when building pipelines.

---

### `agent.result` — recommended accessor (public)

Returns the canonical output of the last call:

- **Typed Pydantic object** — when `output_schema` was active (agent-level or call-level) and parsing succeeded.
- **Plain string** — text content otherwise.
- **`None`** — if the agent has never been called.

```python
from lazybridge import LazyAgent
from pydantic import BaseModel

class BlogPost(BaseModel):
    title: str
    intro: str
    tags: list[str]

writer = LazyAgent("anthropic", output_schema=BlogPost)
writer.chat("Write a post about AI agents")

post = writer.result          # BlogPost instance
print(post.title)             # typed access — no .parsed, no json.loads
print(post.tags)

# Without output_schema — result is a plain string
researcher = LazyAgent("anthropic")
researcher.chat("Find AI news this week")
print(researcher.result)      # str
```

`agent.result` is the recommended access point for pipeline code that needs the final value without caring about the internal representation.

---

### `agent._last_output: str | None` — text-first, for context injection

Always a plain string (`resp.content`). Set after every `chat()`, `loop()`, `achat()`, and `aloop()` call. `None` before the first call.

**Primary consumer:** `LazyContext.from_agent(agent)` reads this field to inject an agent's output into the next agent's system prompt. It is intentionally text-first — system prompts are strings, and injecting a Pydantic object would require explicit serialisation at the call site.

```python
researcher = LazyAgent("anthropic")
researcher.loop("Summarize latest AI papers")

print(researcher._last_output)    # plain text summary

ctx = LazyContext.from_agent(researcher)
# ctx.build() → "[researcher output]\n<summary text>"

writer = LazyAgent("openai", context=ctx)
writer.chat("Write an article based on the research")
```

**Streaming caveat:** on a `chat(stream=True)` call (which returns an iterator, not a `CompletionResponse`), `_last_output` is set *only after the iterator is fully consumed*. If you break out of the loop early, `_last_output` may be incomplete.

---

### `agent._last_response: CompletionResponse | None` — full response (semi-internal)

The complete provider response. Not yet a stable public API — use `agent.result` for the value. Useful for advanced introspection:

| Field | Description |
|---|---|
| `_last_response.content` | Text content (same as `_last_output`) |
| `_last_response.parsed` | Pydantic object (`None` if no output_schema) |
| `_last_response.usage` | `UsageStats(input_tokens, output_tokens)` |
| `_last_response.tool_calls` | List of `ToolCall` objects from the last step |
| `_last_response.grounding_sources` | Citations from native web search |
| `_last_response.thinking` | Extended thinking text (Anthropic only) |

```python
agent = LazyAgent("anthropic", output_schema=BlogPost)
agent.chat("Write a post")

# Use result for the value:
post = agent.result                           # BlogPost

# Use _last_response for metadata:
print(agent._last_response.usage.input_tokens)
print(agent._last_response.usage.output_tokens)
```

---

### Summary — which field to use

| Need | Use |
|---|---|
| The typed or text output of the last call | `agent.result` |
| Inject previous output into next agent's context | `LazyContext.from_agent(agent)` (reads `_last_output`) |
| Token usage / cost accounting | `agent._last_response.usage` |
| Citations from web search | `agent._last_response.grounding_sources` |
| Raw text always | `agent._last_output` |

---

## 9. Async versions

All four methods have async equivalents with identical signatures.

```python
import asyncio
from lazybridge import LazyAgent, LazyTool
from pydantic import BaseModel

async def main():
    ai = LazyAgent("anthropic")

    # achat
    resp = await ai.achat("Hello")
    print(resp.content)

    # aloop
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    add_tool = LazyTool.from_function(add)
    result = await ai.aloop("What is 7 + 8?", tools=[add_tool])
    print(result.content)

    # atext
    text = await ai.atext("What is the capital of Japan?")
    print(text)                     # "Tokyo"

    # ajson
    class Answer(BaseModel):
        value: str

    data = await ai.ajson("Capital of Japan?", schema=Answer)
    print(data.value)               # "Tokyo"

asyncio.run(main())
```

`achat()` with `stream=True` returns an `AsyncIterator[StreamChunk]` — iterate with `async for`.

**Dedicated streaming methods (recommended):**

```python
# Sync — always returns Iterator[StreamChunk]
for chunk in ai.chat_stream("Tell me a story"):
    print(chunk.delta, end="", flush=True)

# Async — always returns AsyncIterator[StreamChunk]
async for chunk in await ai.achat_stream("Tell me a story"):
    print(chunk.delta, end="", flush=True)
```

These are preferred over `chat(stream=True)` because the return type is unambiguous — IDEs can infer it without `isinstance` checks.
