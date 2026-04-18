# Module 1: Your First Agent

## Install

```bash
# Pick your provider(s)
pip install lazybridge[anthropic]
pip install lazybridge[openai]
pip install lazybridge[google]

# Or install all providers at once
pip install lazybridge[all]
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
# or
export GOOGLE_API_KEY="..."
```

## Hello world

```python
from lazybridge import LazyAgent

ai = LazyAgent("anthropic")
response = ai.chat("What is the capital of France?")
print(response.content)
# Paris
```

That's it. One class, one method, one line of real code.

## Understanding the response

`chat()` returns a `CompletionResponse` with useful fields:

```python
resp = ai.chat("What is 2 + 2?")

print(resp.content)        # "4" — the text response
print(resp.usage)          # UsageStats(input_tokens=12, output_tokens=1, ...)
print(resp.stop_reason)    # "end_turn"
print(resp.model)          # "claude-sonnet-4-6"
```

## Shortcut: just get the text

If you only need the text, use `.text()`:

```python
answer = ai.text("What is the capital of France?")
print(answer)  # "Paris" — a plain string, not a response object
```

## Switch providers with one string

```python
# Same code, different provider
ai_openai = LazyAgent("openai")
ai_google = LazyAgent("google")
ai_deepseek = LazyAgent("deepseek")

# They all work the same way
for agent in [ai_openai, ai_google, ai_deepseek]:
    print(agent.text("Say hello in one word"))
```

## System prompts

Tell the agent how to behave:

```python
ai = LazyAgent("anthropic", system="You are a pirate. Always respond in pirate speak.")
print(ai.text("What's the weather like?"))
# Arrr, the skies be clear and the winds be favorable, matey!
```

You can also pass a system prompt per call:

```python
ai = LazyAgent("anthropic")
print(ai.text("What is 2+2?", system="Reply in exactly one word."))
# Four
```

## Choose a specific model

```python
# Use a specific model
ai = LazyAgent("anthropic", model="claude-haiku-4-5-20251001")
ai = LazyAgent("openai", model="gpt-4o-mini")
ai = LazyAgent("google", model="gemini-2.5-flash")
```

Or pick a provider-relative **tier** — `"top"`, `"expensive"`,
`"medium"`, `"cheap"`, or `"super_cheap"` — and let the provider pick
the concrete model:

```python
ai = LazyAgent("anthropic", model="cheap")      # claude-haiku-4-5
ai = LazyAgent("chatgpt",   model="top")        # gpt-5.4
```

Full tier matrix: [Model tiers](../agents.md#model-tiers).

## The `result` property

After any call, `agent.result` gives you the canonical output:

```python
ai = LazyAgent("anthropic")
ai.chat("Hello!")
print(ai.result)  # "Hello! How can I help you today?"
```

This becomes more useful with structured output (Module 3) — `result` returns the typed Pydantic object when available, text otherwise.

## Error handling

```python
from lazybridge import LazyAgent

try:
    ai = LazyAgent("anthropic", api_key="invalid-key")
except ValueError as e:
    print(e)  # "AnthropicProvider requires an API key..."

try:
    ai = LazyAgent("unknown_provider")
except ValueError as e:
    print(e)  # "Unknown provider 'unknown_provider'. Supported: anthropic, claude, ..."
```

## Retry on transient errors

```python
# Automatically retry on 429 (rate limit) and 5xx errors
ai = LazyAgent("anthropic", max_retries=3)
resp = ai.chat("Hello")  # retries up to 3 times with exponential backoff
```

---

## Exercise

1. Create a LazyAgent with your preferred provider
2. Ask it to explain what an LLM is in exactly 3 sentences
3. Print the token usage from the response
4. Try switching to a different provider — does the code change?

**Next:** [Module 2: Tools & Functions](02-tools.md) — give your agent the ability to call your code.
