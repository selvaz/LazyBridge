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
