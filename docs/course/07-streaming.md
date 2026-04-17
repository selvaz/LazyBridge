# Module 7: Streaming

Get tokens as they're generated — essential for chat UIs and long-running responses.

## Basic streaming

```python
from lazybridge import LazyAgent

ai = LazyAgent("anthropic")
for chunk in ai.chat_stream("Write a short poem about coding"):
    print(chunk.delta, end="", flush=True)
print()  # newline at end
```

Each `StreamChunk` has:

- `chunk.delta` — the new text fragment
- `chunk.is_final` — True on the last chunk
- `chunk.stop_reason` — why generation stopped (on final chunk)
- `chunk.usage` — token counts (on final chunk)

## Why `chat_stream()` over `chat(stream=True)`?

```python
# Preferred — return type is always Iterator[StreamChunk]
for chunk in ai.chat_stream("hello"):
    ...

# Legacy — returns CompletionResponse | Iterator[StreamChunk] (union type)
result = ai.chat("hello", stream=True)
# IDE can't tell if result is a response or an iterator
```

`chat_stream()` has an unambiguous return type. Your IDE knows exactly what you're working with.

## Async streaming

```python
import asyncio

async def main():
    ai = LazyAgent("anthropic")
    async for chunk in await ai.achat_stream("Write a haiku"):
        print(chunk.delta, end="", flush=True)
    print()

asyncio.run(main())
```

## Collecting the full response

After consuming the stream, `agent._last_output` has the full text:

```python
for chunk in ai.chat_stream("Tell me a story"):
    print(chunk.delta, end="", flush=True)

print()
print(f"Full response: {ai._last_output}")
print(f"Or use: {ai.result}")
```

**Important:** `_last_output` is only set *after the iterator is fully consumed*. If you `break` early, it may be incomplete.

## Streaming with system prompts

```python
for chunk in ai.chat_stream("Explain gravity", system="Reply in exactly 3 sentences"):
    print(chunk.delta, end="", flush=True)
```

## What can't be streamed

- `loop()` / `aloop()` — tool loops don't support streaming (use `on_event` for progress)
- `json()` / `ajson()` — structured output needs the full response to parse
- `text()` / `atext()` — convenience wrappers, return full text
- `memory=` — needs the full response to record history

---

## Exercise

1. Build a streaming chat loop that prints tokens as they arrive
2. After streaming completes, print the total token count from the final chunk
3. Try streaming with different providers — does the experience differ?

**Next:** [Module 8: Guardrails & Safety](08-guardrails.md) — validate inputs and outputs.
