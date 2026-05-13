# Step 1: What is an LLM?

Before writing a single line of LazyBridge code, you need a mental model of what
you're talking to. This page gives you just enough theory to make the rest of the
tutorial click — no math, no machine learning background required.

---

## LLM = Large Language Model

An LLM is a program that takes text in and produces text out. That's the whole thing.

```
INPUT (prompt)  →  [  LLM  ]  →  OUTPUT (completion)
```

When you use ChatGPT, Claude, or Gemini in a browser, there's an LLM running on a server
somewhere. You type, it replies. Underneath, that's a **text in / text out** system.

---

## The API: talking to an LLM from code

Every major LLM provider (OpenAI, Anthropic, Google) exposes an HTTP API. You send
a JSON payload with your message; you get back a JSON payload with the model's reply.

Here's what that actually looks like — a raw `curl` call to the Anthropic API:

```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-haiku-4-5",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

Response (simplified):

```json
{
  "content": [{"type": "text", "text": "The capital of France is Paris."}],
  "model": "claude-haiku-4-5",
  "usage": {"input_tokens": 15, "output_tokens": 9}
}
```

That's it. The API is a network call. SDKs like `anthropic` or `openai` are just
Python wrappers that save you from writing the `curl` yourself.

---

## Three concepts you must know

### 1. Tokens

LLMs don't read words — they read **tokens**. A token is roughly 3–4 characters, or
¾ of a word. "Hello world" is 2–3 tokens. A full page of text is ~500 tokens.

Why does this matter? Because:

- **You pay per token.** Providers charge for input tokens (your prompt) and output
  tokens (the model's reply). Longer prompts cost more.
- **There's a limit.** Every model has a **context window** — the maximum number of
  tokens it can process in one call. Go over it and the API returns an error.

| Model | Context window | Rough cost (input) |
|---|---|---|
| `claude-haiku-4-5` | 200 K tokens | very cheap |
| `gpt-5.4-mini` | 128 K tokens | cheap |
| `gemini-3-flash-preview` | 1 M tokens | cheap |
| `claude-opus-4-7` | 200 K tokens | expensive |

### 2. Messages (the conversation format)

Every API call passes a list of **messages**. Each message has a `role` and `content`:

```python
messages = [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user",      "content": "And of Germany?"},
]
```

This is how you give the model conversation history — you replay the whole
conversation on every call. The model has no memory between calls; you are the memory.

!!! tip "System prompt"
    There's a special role called `"system"` that sets the model's persona and
    constraints *before* the user speaks. Example: `{"role": "system", "content":
    "You are a concise technical writer. Reply in plain English, no jargon."}`.
    Think of it as the briefing you give a contractor before they start working.

### 3. Temperature

Temperature controls how random the model's output is. Scale: 0.0 → 1.0+.

- `0.0` → deterministic, always the same output for the same input. Use for structured
  data extraction, code generation, classification.
- `0.7` → creative, varied. Use for writing, brainstorming, conversation.
- `1.0+` → chaotic. Usually not useful.

Most agent frameworks default to `0.0`–`0.3` for reliability.

---

## From a single call to an "agent"

A single API call is useful, but limited:

```
User: "Book me a flight to Rome next Tuesday."
LLM:  "I can't do that — I have no access to booking systems."
```

The LLM knows about the world but **can't act on it**. It can only produce text.

An **agent** solves this by giving the LLM **tools** — Python functions the LLM can
call to take real actions:

```
User: "Book me a flight to Rome next Tuesday."

Agent loop:
  turn 1 → LLM decides: call search_flights("Rome", "next Tuesday")
  turn 2 → LLM sees results, decides: call book_flight(flight_id="AZ123")
  turn 3 → LLM produces final answer: "Done! Flight AZ123 confirmed."
```

The framework (LazyBridge in our case) runs this loop: call the LLM, check if it
wants to use a tool, run the tool, feed the result back, repeat until done.

---

## Summary

| Concept | What it means |
|---|---|
| LLM | Text in → text out. Runs on a provider's server. |
| API call | Send JSON with your messages, get JSON with the reply. |
| Token | Unit of text the model reads. You pay per token. |
| Context window | Max tokens per call. Older context gets dropped when you exceed it. |
| System prompt | Persona/constraints set before the conversation starts. |
| Temperature | How random the output is. Lower = more deterministic. |
| Agent | LLM + tools + a loop that runs until the task is done. |

---

## Next step

In the next page you'll see what calling these APIs looks like in raw Python —
with the official OpenAI, Anthropic, and Gemini SDKs — before we switch to
LazyBridge and see how much simpler it gets.

[**Step 2: Raw SDKs compared →**](02-raw-sdks.md){ .md-button .md-button--primary }

[← Back to Start](index.md){ .md-button }
