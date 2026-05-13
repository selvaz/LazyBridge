# Step 2: Raw SDKs compared

Now that you know what an LLM is, let's write some actual Python.

We'll solve the same simple task four times:

> **Task:** Ask an LLM to summarise a short text in three bullet points.

First with the three major raw SDKs, then with LazyBridge. By the end you'll see
exactly what LazyBridge removes — and why that matters as soon as your project
grows beyond a single call.

---

## The text we'll summarise

```python
TEXT = """
Quantum computing uses quantum-mechanical phenomena such as superposition and
entanglement to process information. Unlike classical bits, which are either 0
or 1, quantum bits (qubits) can exist in both states simultaneously. This
parallelism allows quantum computers to solve certain problems — like
cryptography and drug discovery — exponentially faster than classical machines.
As of 2026, quantum computers remain largely experimental, but companies like
IBM, Google, and startups worldwide are racing to reach practical quantum
advantage.
"""
```

---

## Option A — OpenAI SDK (Responses API)

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI

client = OpenAI()                          # reads OPENAI_API_KEY from env

response = client.responses.create(
    model="gpt-5.4-mini",
    instructions="You are a concise technical writer.",
    input=f"Summarise this in exactly 3 bullet points:\n\n{TEXT}",
    max_output_tokens=256,
    temperature=0.3,
)

summary = response.output_text             # convenience accessor
print(summary)
```

**What you have to manage yourself:**

- `input=` accepts a string for one-shot calls, but for a multi-turn conversation you
  switch to a list of message dicts (`[{"role": "user", "content": "..."}, ...]`) — two
  different shapes in the same parameter
- Stateful conversations require either replaying the full history every call, or
  threading `previous_response_id` between calls — your responsibility either way
- Handling `RateLimitError`, `APIConnectionError`, timeouts — no built-in retry
- If you want the model to call a function, you must write a JSON schema by hand and
  parse the `tool_call` items out of `response.output` yourself
- If you want streaming, you rewrite the whole call with `stream=True` and iterate events

!!! note "Responses API vs Chat Completions"
    OpenAI's `responses.create()` is the recommended endpoint as of 2025 — it
    supports stateful conversations, built-in tools (web search, code interpreter),
    and structured output natively. The older `chat.completions.create()` still
    works for legacy code, but new projects should default to Responses.

---

## Option B — Anthropic SDK

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
import anthropic

client = anthropic.Anthropic()             # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    system="You are a concise technical writer.",   # system is a top-level param here
    messages=[
        {
            "role": "user",
            "content": f"Summarise this in exactly 3 bullet points:\n\n{TEXT}",
        }
    ],
)

summary = response.content[0].text        # note: different path from OpenAI
print(summary)
```

**What changes vs OpenAI:**

- `system=` is a top-level parameter, not a message — different shape
- Response lives at `response.content[0].text`, not `.choices[0].message.content`
- Tool use requires a different dict structure (`"type": "tool_use"` blocks)
- Retry / timeout handling: again, your problem

!!! warning "Different SDKs, different shapes"
    If you ever want to switch from OpenAI to Anthropic (or vice versa), you rewrite
    every call site. The response shapes are incompatible; the error types are
    incompatible; the tool-calling format is incompatible. You end up maintaining an
    abstraction layer yourself — which is exactly what LazyBridge is.

---

## Option C — Google Gemini SDK

```bash
pip install google-genai
export GEMINI_API_KEY="AIza..."
```

```python
from google import genai
from google.genai import types

client = genai.Client()                   # reads GEMINI_API_KEY from env

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction="You are a concise technical writer.",
        max_output_tokens=256,
        temperature=0.3,
    ),
    contents=f"Summarise this in exactly 3 bullet points:\n\n{TEXT}",
)

summary = response.text                   # yet another response shape
print(summary)
```

**What changes vs OpenAI and Anthropic:**

- Third different API shape — `contents=` instead of `messages=`
- `config=` object instead of flat kwargs
- Response at `.text` (simpler, but still different)
- Switching Gemini model versions can change the config structure

---

## Now — with LazyBridge

```bash
pip install lazybridge
export ANTHROPIC_API_KEY="sk-ant-..."    # or OPENAI_API_KEY, or GEMINI_API_KEY
```

```python
from lazybridge import Agent, LLMEngine

agent = Agent(engine=LLMEngine("claude-haiku-4-5", system="You are a concise technical writer."))
summary = agent(f"Summarise this in exactly 3 bullet points:\n\n{TEXT}").text()
print(summary)
```

Three lines. That's it.

**Switch to OpenAI? Change one string:**

```python
agent = Agent(engine=LLMEngine("gpt-5.4-mini", system="You are a concise technical writer."))
```

**Switch to Gemini:**

```python
agent = Agent(engine=LLMEngine("gemini-3-flash-preview", system="You are a concise technical writer."))
```

The rest of your code stays identical.

!!! tip "Reading the call"
    `Agent(engine=LLMEngine(...))` is the canonical form. `LLMEngine` wraps the model
    + its config (`system=`, `max_turns=`, `temperature=`, ...). `Agent` adds the
    runtime around it: tool dispatch, retries, observability.

    Calling `agent(prompt)` runs the agent; `.text()` extracts the final string from
    the result envelope. (The envelope also carries cost, tokens, latency, and any
    typed payload — see Step 3.)

---

## Side-by-side comparison

| | OpenAI SDK | Anthropic SDK | Gemini SDK | LazyBridge |
|---|:---:|:---:|:---:|:---:|
| Lines of code (simple call) | ~15 | ~15 | ~15 | **3** |
| Unified response shape | ✗ | ✗ | ✗ | ✓ `.text()` |
| Provider switch = 1 string | ✗ | ✗ | ✗ | ✓ |
| Retry on transient errors | ✗ | ✗ | ✗ | ✓ built-in |
| Tool calling (no JSON schema) | ✗ | ✗ | ✗ | ✓ from type hints |
| Streaming | manual | manual | manual | ✓ `agent.stream()` |
| Structured output (Pydantic) | manual | manual | manual | ✓ `output=MyModel` |
| Multi-agent orchestration | ✗ | ✗ | ✗ | ✓ |

---

## What LazyBridge does NOT replace

LazyBridge wraps the official SDKs — it doesn't replace them. If you need access to
a very new, SDK-specific feature that LazyBridge hasn't exposed yet, you can always
drop down to the raw SDK. LazyBridge is an **addition** to the ecosystem, not a wall.

---

## The mock variant (no API key needed)

If you don't have a key yet, you can run this with `MockAgent` — a test double
that returns a fixed response without making any network call:

```python
from lazybridge.testing import MockAgent

agent = MockAgent(
    ["• Quantum computers use qubits instead of classical bits.\n"
     "• Qubits can be 0, 1, or both simultaneously (superposition).\n"
     "• This enables exponentially faster solutions for specific problems."],
    name="summariser",
)

summary = agent(f"Summarise this in exactly 3 bullet points:\n\n{TEXT}").text()
print(summary)
```

Same `.text()` interface — your downstream code doesn't change when you swap
`MockAgent` for a real `Agent`.

---

## Summary

| SDK | Good for |
|---|---|
| `openai` | Direct OpenAI feature access; fine for a one-file script |
| `anthropic` | Direct Anthropic feature access; fine for a one-file script |
| `google-genai` | Direct Gemini feature access; fine for a one-file script |
| **LazyBridge** | Multi-provider projects, tools, agents, production code |

The raw SDKs are the right choice for a one-off script that only ever calls one
provider and never needs tools. For everything else — and especially once you add
tools, structure your output, or orchestrate multiple agents — LazyBridge saves
significant boilerplate and keeps your code readable.

---

[**Step 3: Your first agent →**](03-first-agent.md){ .md-button .md-button--primary }

[← Step 1: What is an LLM?](01-what-is-an-llm.md){ .md-button }
