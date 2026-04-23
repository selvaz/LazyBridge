# Recipe: Structured output

**Tier:** Basic  
**Goal:** Get a typed Pydantic model instance back instead of plain text.

Pass `output=YourModel` to `Agent`. The returned `Envelope.payload` is a validated instance
of your model. Use `.payload` for typed access; `.text()` gives the JSON dump.

## Minimal case

```python
from lazybridge import Agent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

env = Agent("claude-opus-4-7", output=Article)("write a short article about Python async")

# .payload is a typed Article instance — the static checker knows this
print(env.payload.title)
print(env.payload.tags)

# .text() returns the JSON dump — use .payload for field access
```

## Structured output with tools

Tools and structured output work together. The model calls tools to gather data,
then returns a structured response:

```python
from lazybridge import Agent
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temp_c: float
    conditions: str
    advice: str

def get_weather(city: str) -> str:
    """Return temperature and conditions for ``city``."""
    return f"{city}: 22°C, sunny"

env = Agent("claude-opus-4-7", tools=[get_weather], output=WeatherReport)(
    "weather report for Paris"
)
report = env.payload
print(f"{report.city}: {report.temp_c}°C — {report.advice}")
```

## Checking success before reading

`Envelope.ok` is `True` when no error occurred. Check it before reading `.payload` in
production code:

```python
env = Agent("claude-opus-4-7", output=Article)("write an article")

if env.ok:
    print(env.payload.title)
else:
    print(f"failed ({env.error.type}): {env.error.message}")
```

## Routing via `next` field

Add a `next: Literal[...]` field to your model and `Plan` uses it to route to the
matching step — no extra wiring needed:

```python
from pydantic import BaseModel
from typing import Literal

class SearchResult(BaseModel):
    items: list[str]
    next: Literal["process", "no_results"] = "process"
    # Plan will route to the "process" or "no_results" step based on this value
```

See [Plan with resume](plan-with-resume.md) for a full routing example.

## Next

- [Plan with resume](plan-with-resume.md) — multi-step typed pipeline with routing
- [verify=](../guides/verify.md) — add a judge/retry loop to validate the output
- [Envelope](../guides/envelope.md) — full `Envelope` and `EnvelopeMetadata` reference
