# Step 3: Your first agent

In Step 2 you saw the same task in four shapes — three raw SDKs and
LazyBridge — to compare boilerplate. This step zooms in on **the
LazyBridge side**: what an `Agent` actually is, what it returns, and
how to give it the capabilities you'll keep reaching for (a persistent
persona, observability, structured output, a stable name).

We'll build the same agent four times, adding one capability at a time,
so each new concept appears next to working code.

!!! note "On 'agent' definitions"
    Step 1 defined an *agent* informally as "LLM + tools + a loop". In
    LazyBridge the **class** `Agent` is broader: it's the standard
    container for any LLM-powered task — tools optional. The "tools
    loop" only kicks in when you actually pass `tools=[...]` (Step 4).
    Everything in this step uses an `Agent` *without* tools and still
    benefits from the framework (typed output, observability,
    verification, multi-agent composition).

---

## The starting point

The smallest valid LazyBridge program:

```python
from lazybridge import Agent, LLMEngine

agent = Agent(engine=LLMEngine("claude-haiku-4-5"))
result = agent("What is the capital of France?")
print(result.text())
```

Three things to notice:

- **`Agent(engine=LLMEngine(...))`** — `LLMEngine` configures the model; `Agent` wraps
  it with the runtime (retries, tool dispatch, observability)
- **`agent(prompt)`** — calling the agent like a function runs one task
- **`.text()`** — extracts the final string from the result

That last point matters more than it looks. Let's unpack it.

---

## Don't pick a model — pick a tier

You don't have to memorise model names. LazyBridge has a small set of
**tier aliases** that each provider maps to its current best SKU. Use them
in everyday code; pin a specific model string only when an example tests a
model-specific behaviour.

```python
from lazybridge import Agent

agent = Agent.from_provider("anthropic", tier="top")           # smartest Anthropic model
result = agent("What is the capital of France?")
print(result.text())
```

`Agent.from_provider` is shorthand for `Agent(engine=LLMEngine(<resolved-model>))`.
The tier name expands at construction time — *and stays current as
providers ship new SKUs*. You don't rewrite your code when Anthropic
releases the next Opus.

The five tier aliases, ordered cheapest → smartest:

| Tier | Use when | Anthropic | OpenAI | Google | DeepSeek |
|---|---|---|---|---|---|
| `super_cheap` | High-volume classification, simple extraction | (uses `cheap`) | `gpt-5.4-nano` | (uses `cheap`) | `deepseek-v4-flash` |
| `cheap` | Tools dispatch, summaries, drafts | `claude-haiku-4-5` | `gpt-5.4-nano` | `gemini-3-flash-preview` | `deepseek-v4-flash` |
| `medium` *(default)* | Most agent work — sensible all-rounder | `claude-haiku-4-5` | `gpt-5.4-mini` | `gemini-3-flash-preview` | `deepseek-v4-flash` |
| `expensive` | Complex reasoning, long-context work | `claude-opus-4-7` | `gpt-5.4` | `gemini-3.1-pro-preview` | `deepseek-v4-pro` |
| `top` | Best available — production-critical, hard problems | `claude-opus-4-7` | `gpt-5.4-pro` | `gemini-3.1-pro-preview` | `deepseek-v4-pro` |

Use them like this:

```python
draft_agent  = Agent.from_provider("anthropic", tier="cheap")    # bulk drafts
final_agent  = Agent.from_provider("anthropic", tier="top")      # final pass
multi_model  = Agent.from_provider("openai",    tier="medium")   # one-string swap to OpenAI
```

!!! tip "When to pin a specific model id"
    Tier aliases follow each provider's current SKU. **Pin** a specific
    model string (e.g. `"claude-opus-4-7"`) only when:

    - You're writing a recipe that needs reproducible behaviour for a
      specific model
    - You're testing a model-specific feature (e.g. extended thinking on
      `claude-opus-4-7`)
    - Compliance / audit requires a frozen model id

    For everything else — *especially your own everyday code* — use
    `tier=`. It's the beginner-friendly default and the production-friendly
    one too.

In the rest of this tutorial we'll mix both forms: tier aliases for code
you'd write daily, pinned model strings when the example illustrates a
specific point.

---

## What `result` actually is — the Envelope

`agent(...)` does not return a plain string. It returns an **`Envelope`** — an object
that carries the answer *plus* everything you might need to know about the run:

```python
result = agent("What is the capital of France?")

result.text()              # "The capital of France is Paris."
result.payload             # same string (str by default — Pydantic model when output= is set)
result.error               # None if successful; the raised exception otherwise
result.ok                  # True / False
result.metadata.cost_usd   # 0.00012
result.metadata.input_tokens
result.metadata.output_tokens
result.metadata.latency_ms
result.metadata.model      # "claude-haiku-4-5"
result.metadata.provider   # "anthropic"
result.metadata.run_id     # UUID for log correlation
```

This is one of LazyBridge's core decisions: **every call returns the same shape**,
regardless of which provider you used or whether the agent invoked tools, ran sub-agents,
or executed a plan. Your downstream code never has to special-case "single call vs.
multi-agent run".

```python
# Quick inspection
print(f"Answer:  {result.text()}")
print(f"Cost:    ${result.metadata.cost_usd:.4f}")
print(f"Tokens:  {result.metadata.input_tokens} in / {result.metadata.output_tokens} out")
print(f"Latency: {result.metadata.latency_ms} ms")
```

!!! tip "Why an envelope and not a tuple?"
    Raw SDKs return different objects (`response.choices[0].message.content` vs
    `response.content[0].text` vs `response.output_text`). When you switch providers
    you rewrite every call site. The Envelope is the contract that lets you change
    `LLMEngine("claude-haiku-4-5")` → `LLMEngine("gpt-5.4-mini")` without touching
    anything else.

---

## Giving the agent a personality — `system=`

By default the model has no persona. To give it one, set `system=` on the engine:

```python
from lazybridge import Agent, LLMEngine

agent = Agent(engine=LLMEngine(
    "claude-haiku-4-5",
    system="You are a witty Roman tour guide. Reply in 1-2 sentences, in English.",
))

print(agent("What is the capital of France?").text())
# > "Paris, of course — though as a proud Roman, I'll add that Rome was the
# > original 'Eternal City' centuries before Paris was even a Gaulish village."
```

The system prompt is **stable across every call** to that agent. Use it for:

- Persona and tone ("concise technical writer", "supportive coach")
- Hard rules ("never reveal internal IDs", "always reply in JSON")
- Domain framing ("you are reviewing a Python pull request")

The user prompt — what you pass into `agent(...)` — changes each call. The system
prompt is the briefing.

---

## Seeing inside the run — `verbose=True`

When your agent does more than one thing, you'll want to see what's happening.
`verbose=True` prints a turn-by-turn trace to stdout:

```python
agent = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You are a concise math tutor."),
    verbose=True,
)

agent("Explain step by step: what is 15% of 240?")
```

Output (abbreviated):

```text
[agent ▶ engine=LLMEngine model=claude-haiku-4-5]
  user: Explain step by step: what is 15% of 240?
  assistant: Step 1: Convert 15% to 0.15
             Step 2: Multiply: 0.15 × 240 = 36
             Answer: 36
[done] turns=1  tokens=42/58  cost=$0.00008  latency=412ms
```

You won't ship code with `verbose=True` — but it's the fastest way to debug an
agent during development. Step 4 will show where this gets *really* useful: once
the agent starts calling tools, `verbose` shows every tool call and result.

---

## Getting a typed answer — `output=`

So far the agent returns free-form text. For production code you usually want
*structured* data — a Python object with named fields you can read directly.

LazyBridge takes a Pydantic model and forces the LLM's response through it:

```python
from pydantic import BaseModel, Field
from lazybridge import Agent, LLMEngine


class CapitalInfo(BaseModel):
    city: str = Field(..., description="The capital city")
    country: str = Field(..., description="The country it's the capital of")
    population_millions: float = Field(..., description="Approximate population, in millions")


agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    output=CapitalInfo,
)

result = agent("What is the capital of France?")
info: CapitalInfo = result.payload          # typed object, not a string

print(info.city)                # "Paris"
print(info.country)             # "France"
print(info.population_millions) # 2.16
```

Two things happened under the hood:

1. LazyBridge auto-generated a JSON schema from your Pydantic model and passed it
   to the provider's structured-output mode
2. The model's response was parsed and validated against `CapitalInfo` *before*
   you got the envelope; if validation fails, LazyBridge retries with feedback
   (up to `max_output_retries`, default 2)

`result.text()` still returns a string (the JSON), but `result.payload` is the
real value: a Python object with type hints your IDE understands.

!!! warning "Raw SDKs make you do this yourself"
    With the raw OpenAI / Anthropic / Gemini SDKs you'd write the JSON schema by
    hand, pass it as `response_format=` (or equivalent), parse `response.text`
    with `json.loads`, and write your own retry-on-validation-failure loop.
    That's 40+ lines you don't write here.

---

## Naming your agent — `name=`

You can attach a name to any agent:

```python
researcher = Agent(
    engine=LLMEngine("claude-haiku-4-5", system="You research facts on demand."),
    name="researcher",
)
```

The name does nothing for a standalone agent — but in Step 5 you'll see it become
the **stable identifier** that lets agents reference each other (`Step("researcher")`,
`from_agent("researcher")`, `tools=[researcher]`). Get into the habit of naming
agents now and the multi-agent step will feel natural.

---

## Putting it all together

A practical first agent — system prompt, structured output, observability:

```python
from pydantic import BaseModel, Field
from lazybridge import Agent, LLMEngine


class MovieReview(BaseModel):
    title: str
    rating: int = Field(..., ge=1, le=10, description="1-10 stars")
    one_line_verdict: str
    tags: list[str] = Field(..., max_length=5, description="Up to 5 genre/mood tags")


reviewer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="You are a sharp, fair film critic. Be specific, not vague.",
    ),
    output=MovieReview,
    name="reviewer",
)

result = reviewer("Write a short review of Blade Runner 2049.")

review: MovieReview = result.payload
print(f"{review.title} — {review.rating}/10")
print(f"  {review.one_line_verdict}")
print(f"  tags: {', '.join(review.tags)}")
print(f"\n[cost ${result.metadata.cost_usd:.4f}, "
      f"{result.metadata.input_tokens}+{result.metadata.output_tokens} tokens]")
```

Sample output:

```text
Blade Runner 2049 — 9/10
  A patient, gorgeous sequel that earns its 163 minutes by trusting the audience.
  tags: sci-fi, neo-noir, slow-burn, visual, philosophical

[cost $0.0021, 218+184 tokens]
```

---

## Structured output — raw SDKs vs LazyBridge

Since structured output is one of the biggest practical wins, here's the same
`CapitalInfo` task with each major SDK so you can see what LazyBridge actually
spares you.

The Pydantic model is identical across all four:

```python
from pydantic import BaseModel, Field

class CapitalInfo(BaseModel):
    city: str
    country: str
    population_millions: float = Field(..., description="Approximate, in millions")
```

### OpenAI (Responses API, `.parse()` helper)

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.parse(
    model="gpt-5.4-mini",
    instructions="Return capital info.",
    input="What is the capital of France?",
    text_format=CapitalInfo,            # OpenAI-specific kwarg
)

info: CapitalInfo = response.output_parsed
```

What you still own: retry on validation failure (the `.parse()` helper raises if
the model deviates from the schema), provider-specific error handling, switching
to a different provider means rewriting the call.

### Anthropic (`.parse()` helper)

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.parse(
    model="claude-haiku-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    output_format=CapitalInfo,           # Anthropic-specific kwarg
)

info: CapitalInfo = response.parsed_output
```

What you still own: retry on validation failure, provider-specific error handling,
switching to a different provider means rewriting the call.

!!! note "Tool-call workaround is no longer canonical"
    Before native structured output landed in the Anthropic SDK, the pattern was to
    force a tool call with the schema as its input and parse the `tool_use` block
    by hand (~13 lines of boilerplate). Many older tutorials still teach that.
    If you see a `tool_choice={"type": "tool", ...}` example with manual
    `model_validate()`, it's pre-`.parse()` code — use `messages.parse()` /
    `output_format=` instead.

### Gemini

Gemini does have native structured output via `response_schema=`:

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="What is the capital of France?",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CapitalInfo,
    ),
)

info: CapitalInfo = response.parsed
```

What you still own: the `config=` object, the mime-type string, switching providers
means rewriting the call.

### LazyBridge

```python
from lazybridge import Agent, LLMEngine

agent = Agent(engine=LLMEngine("claude-haiku-4-5"), output=CapitalInfo)
info: CapitalInfo = agent("What is the capital of France?").payload
```

One kwarg: `output=CapitalInfo`. Provider auto-routed; retry on validation
failure happens automatically (default `max_output_retries=2`); the *same code*
works on Claude, GPT, Gemini, DeepSeek, or any custom provider — change the
model string and that's it.

### Side-by-side cost

| | OpenAI Responses | Anthropic | Gemini | LazyBridge |
|---|:---:|:---:|:---:|:---:|
| Lines to set up | ~7 | ~7 | ~9 | **2** |
| Manual schema generation | hidden | hidden | hidden | hidden |
| Manual response parsing | no (`.parse()`) | no (`.parse()`) | no (`.parsed`) | no (`.payload`) |
| Built-in validation retry | no | no | no | **yes** |
| Provider switch | rewrite call | rewrite call | rewrite call | change one string |



| Concept | Syntax | What it gives you |
|---|---|---|
| Build an agent (pinned model) | `Agent(engine=LLMEngine("..."))` | Use when you need a specific SKU |
| Build an agent (tier) | `Agent.from_provider("anthropic", tier="top")` | Beginner-friendly; stays current as providers ship new SKUs |
| System prompt | `LLMEngine("...", system="...")` | Stable persona / rules |
| Run it | `agent(prompt)` | Returns an `Envelope` |
| Get the text | `result.text()` | Final assistant string |
| Get metrics | `result.metadata.cost_usd`, etc. | Cost, tokens, latency, model |
| Trace the run | `Agent(..., verbose=True)` | Turn-by-turn stdout output |
| Typed answer | `Agent(..., output=MyModel)` | `result.payload` is your Pydantic model |
| Name it | `Agent(..., name="researcher")` | Stable handle for multi-agent setups |

You now have a single, well-behaved agent. The next step is the one that makes
agents *interesting*: giving them the ability to **call functions in your codebase**.

---

[**Step 4: Giving your agent tools →**](04-tools.md){ .md-button .md-button--primary }

[← Step 2: Raw SDKs compared](02-raw-sdks.md){ .md-button }
