# Step 3: Your first agent

In Step 2 you called an LLM once. That's already useful — but it's still just a wrapper
around a single API call. In this step you'll learn what makes an `Agent` an *agent*,
and why every result is wrapped in an `Envelope`.

We'll build the same agent four times, adding one capability at a time, so each new
concept appears next to working code.

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

### Anthropic (tool-call workaround)

Anthropic has no native "give me a Pydantic object" endpoint. The canonical
pattern is forcing a tool call with the schema as its input:

```python
import anthropic
from pydantic import BaseModel

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,
    tools=[{
        "name": "record_capital_info",
        "description": "Record capital city info.",
        "input_schema": CapitalInfo.model_json_schema(),   # generate the schema
    }],
    tool_choice={"type": "tool", "name": "record_capital_info"},
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

# Find the tool_use block in response.content and validate manually
tool_block = next(b for b in response.content if b.type == "tool_use")
info: CapitalInfo = CapitalInfo.model_validate(tool_block.input)
```

What you still own: writing the synthetic tool, the `tool_choice` boilerplate,
finding the right content block, manual `model_validate()`, retry-on-failure.

### Gemini

Gemini does have native structured output via `response_schema=`:

```python
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.0-flash",
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
| Lines to set up | ~7 | ~13 | ~9 | **2** |
| Manual schema generation | hidden | yes (`.model_json_schema()`) | hidden | hidden |
| Manual response parsing | no (`.parse()`) | yes | no (`.parsed`) | no (`.payload`) |
| Built-in validation retry | no | no | no | **yes** |
| Provider switch | rewrite call | rewrite call | rewrite call | change one string |



| Concept | Syntax | What it gives you |
|---|---|---|
| Build an agent | `Agent(engine=LLMEngine("..."))` | A callable LLM wrapper |
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
