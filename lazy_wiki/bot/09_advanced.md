# Advanced Features

## Structured Output

Force the model to return valid JSON matching a schema. Use a Pydantic model or a raw JSON Schema dict.

```python
from pydantic import BaseModel
from lazybridge import LazyAgent

class ArticleIdeas(BaseModel):
    title: str
    summary: str
    keywords: list[str]

ai = LazyAgent("anthropic")

# Returns the Pydantic instance directly in resp.parsed
resp = ai.chat("Generate an article idea about AI safety", output_schema=ArticleIdeas)
idea: ArticleIdeas = resp.parsed
print(idea.title)

# Shortcut: returns parsed directly
idea = ai.json("Generate an article idea about AI safety", ArticleIdeas)

# Async
idea = await ai.ajson("Generate an article idea about AI safety", ArticleIdeas)
```

With raw JSON Schema dict:
```python
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "score": {"type": "number"}
    },
    "required": ["sentiment", "score"]
}
result = ai.json("Analyse sentiment: 'This is great!'", schema)
print(result["sentiment"])  # "positive"
```

On parse failure: `resp.validated == False`, `resp.validation_error` contains the error string. Call `resp.raise_if_failed()` to surface it as an exception.

```python
from lazybridge import LazyAgent, StructuredOutputError

ai = LazyAgent("anthropic")

resp = ai.chat("Generate an article idea about AI safety", output_schema=ArticleIdeas)
if resp.validated:
    idea: ArticleIdeas = resp.parsed
else:
    print(resp.validation_error)   # human-readable reason

# Or let it raise:
try:
    resp.raise_if_failed()
except StructuredOutputError as exc:
    # StructuredOutputParseError      — JSON could not be parsed
    # StructuredOutputValidationError — JSON parsed but failed schema/Pydantic validation
    print(type(exc).__name__, exc)
```

`StructuredOutputError` is the common base class. `StructuredOutputParseError` and `StructuredOutputValidationError` are the concrete subclasses raised by `raise_if_failed()`.

---

## Extended Thinking

Enable chain-of-thought reasoning before answering.

```python
from lazybridge.core.types import ThinkingConfig

ai = LazyAgent("anthropic", model="claude-sonnet-4-6")

# Simple boolean toggle
resp = ai.chat("Solve: what is 17 * 23?", thinking=True)
print(resp.thinking)   # the internal reasoning
print(resp.content)    # the final answer

# Full config
thinking_cfg = ThinkingConfig(
    enabled=True,
    effort="high",       # "low" | "medium" | "high" | "xhigh"
    display="omitted",   # Anthropic 4.6+: omit thinking text from stream output
)
resp = ai.chat("Complex analysis task", thinking=thinking_cfg)
```

---

## Streaming

```python
# Sync streaming
for chunk in ai.chat("Tell me a story", stream=True):
    print(chunk.delta, end="", flush=True)
    if chunk.is_final:
        print()
        print(f"Tokens used: {chunk.usage.output_tokens}")

# Async streaming
async for chunk in await ai.achat("Tell me a story", stream=True):
    print(chunk.delta, end="", flush=True)
```

`StreamChunk` fields:
- `delta: str` — new text in this chunk
- `thinking_delta: str` — new thinking text (if thinking enabled)
- `tool_calls: list[ToolCall]` — populated on is_final if tools used
- `stop_reason: str | None` — populated on is_final
- `usage: UsageStats | None` — populated on is_final
- `is_final: bool` — last chunk in the stream
- `parsed: Any` — populated on is_final if output_schema was set
- `grounding_sources: list[GroundingSource]` — populated on is_final

---

## Native Tools (Provider-Side)

Provider-managed tools (e.g. web search, code execution). Run on the provider's infrastructure — no local callable needed.

```python
from lazybridge.core.types import NativeTool

ai = LazyAgent("anthropic")
resp = ai.chat("What is the current price of Bitcoin?", native_tools=[NativeTool.WEB_SEARCH])

# String aliases also work
resp = ai.chat("Search for Python 3.13 release notes", native_tools=["web_search"])
```

Available native tools:
- `NativeTool.WEB_SEARCH` — Anthropic, OpenAI, Google
- `NativeTool.CODE_EXECUTION` — Anthropic, OpenAI
- `NativeTool.FILE_SEARCH` — OpenAI
- `NativeTool.COMPUTER_USE` — Anthropic
- `NativeTool.GOOGLE_SEARCH` — Google Gemini grounding
- `NativeTool.GOOGLE_MAPS` — Google Gemini

Grounding citations in response:
```python
for src in resp.grounding_sources:
    print(src.url, src.title, src.snippet)
```

---

## Skills (Anthropic Only)

Server-side domain-expert packages for document processing.

```python
ai = LazyAgent("anthropic")
resp = ai.chat(
    "Summarise this PDF: [upload the file via your interface]",
    skills=["pdf", "excel", "powerpoint", "word"],
)
```

---

## Retry with Backoff

Configure automatic retry on transient API errors (429, 5xx).

```python
ai = LazyAgent("anthropic", max_retries=3)
# Retries up to 3 times with exponential backoff + ±10% jitter
# Delay formula: base_delay * 2^attempt * random(0.9, 1.1)
```

Default `max_retries=0` (no retry).

---

## Multi-provider agent (provider swap)

Changing provider requires only changing the first argument. The API is identical across providers.

```python
# Same code, different provider
ai_claude  = LazyAgent("anthropic", model="claude-sonnet-4-6")
ai_gpt     = LazyAgent("openai",    model="gpt-4o")
ai_gemini  = LazyAgent("google",    model="gemini-2.0-flash")
ai_deepseek = LazyAgent("deepseek")

# All use identical API
for ai in [ai_claude, ai_gpt, ai_gemini, ai_deepseek]:
    print(ai.text("What is 2+2?"))
```

---

## tool.specialize() — role-specific tool variants

```python
from lazybridge import LazyTool

def search(query: str, max_results: int = 5) -> str:
    """Search the web."""
    ...

base = LazyTool.from_function(search)

# Specialise for different contexts
us_search = base.specialize(
    name="search_us_news",
    guidance="Always restrict results to US news sources published in the last 24 hours."
)
eu_search = base.specialize(
    name="search_eu_news",
    guidance="Always restrict results to EU news sources published in the last 24 hours."
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop("Compare today's AI news in the US vs EU", tools=[us_search, eu_search])
```

---

## Pipeline Tools — `LazyTool.parallel()` / `LazyTool.chain()`

Session-free factory methods that compose agents and tools into a single `LazyTool`. No `LazySession` required. `LazySession.as_tool(mode=...)` is a thin wrapper over these — semantically identical.

### `parallel()` — fan-out

All participants run concurrently on the same task. Results are combined (default: concatenated with agent-name headers).

```python
from lazybridge import LazyAgent, LazyTool

us     = LazyAgent("anthropic", name="us",     system="Report US AI news.")
europe = LazyAgent("openai",    name="europe",  system="Report European AI news.")
asia   = LazyAgent("google",    name="asia",    system="Report Asian AI news.")

news_tool = LazyTool.parallel(
    us, europe, asia,
    name="world_news",
    description="Parallel AI news summary from US, Europe, and Asia",
    combiner="concat",          # "concat" (default) | "last"
    concurrency_limit=3,        # max simultaneous API calls; None = all at once
    step_timeout=30.0,          # per-participant timeout in seconds; None = no timeout
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop("Produce a global AI news digest.", tools=[news_tool])
```

**`concurrency_limit`:** caps the number of participants running at the same time using `asyncio.Semaphore`. Use this when hitting API rate limits or when participants share a scarce resource. `None` (default) fires all coroutines simultaneously.

**`step_timeout`:** wraps each participant coroutine with `asyncio.wait_for(coro, timeout=step_timeout)`. Timed-out participants return `"[ERROR: TimeoutError: ...]"` in `concat` mode (captured via `return_exceptions=True` in `asyncio.gather`). In `last` mode they propagate as `TimeoutError`.

**Cloning:** participants are cloned per invocation. After the run, `us._last_output` is `None` — use the tool's return value or the orchestrator's response.

### `chain()` — sequential handoff

Participants run in order. Each step passes its output to the next.

**Async-under-the-hood:** `chain()` uses `build_achain_func` — every step calls `achat()` / `aloop()` / `ajson()`, so the event loop is never blocked. `run()` drives it via `run_async()`; `arun()` awaits it directly.

```python
from lazybridge import LazyAgent, LazyTool

researcher  = LazyAgent("anthropic", name="researcher",  system="Research AI topics in depth.")
summariser  = LazyAgent("openai",    name="summariser",  system="Summarise research concisely.")
fact_checker = LazyAgent("anthropic", name="checker",    system="Verify factual claims.")

pipeline = LazyTool.chain(
    researcher, summariser, fact_checker,
    name="research_pipeline",
    description="Research, summarise, and fact-check a topic.",
    step_timeout=60.0,          # per-step timeout in seconds; asyncio.TimeoutError raised on breach
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop("Produce a verified report on fusion energy.", tools=[pipeline])
```

**`step_timeout`:** wraps each step with `asyncio.wait_for(step_coro, timeout=step_timeout)`. Unlike parallel, a timeout in chain raises `asyncio.TimeoutError` immediately (no gathering). Use to prevent a hanging step from blocking the whole pipeline.

**Handoff semantics:**

| Previous step | Next step receives |
|---|---|
| Agent | Original task + previous agent's output injected as context |
| Tool | Tool's output becomes the new task directly |

### Cross-session validation

Pass `session=` to validate that all participants belong to (or are compatible with) the same session. Checked at creation time, not at run time.

```python
# Raises ValueError if any participant is bound to a different session
pipeline = LazyTool.parallel(a, b, name="...", description="...", session=my_session)
```

This also covers `LazyTool.from_agent()` delegate tools — the inner agent's session is checked.

### `save()` restriction

`parallel()` and `chain()` tools have `_is_pipeline_tool = True`. Calling `save()` raises `ValueError` — they are runtime compositions and cannot be serialized. Save individual participants via `agent.as_tool().save()` instead.
