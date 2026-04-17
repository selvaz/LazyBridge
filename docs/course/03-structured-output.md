# Module 3: Structured Output

Instead of parsing text responses manually, get typed Pydantic objects directly from the LLM.

## Basic structured output

```python
from pydantic import BaseModel
from lazybridge import LazyAgent

class City(BaseModel):
    name: str
    country: str
    population: int

ai = LazyAgent("anthropic")
city = ai.json("Tell me about Tokyo", City)

print(city.name)        # "Tokyo"
print(city.country)     # "Japan"
print(city.population)  # 13960000
print(type(city))       # <class 'City'>
```

`json()` returns a validated Pydantic instance — not a dict, not a string.

## Complex schemas

Pydantic models can be nested, have lists, optionals, enums:

```python
from enum import Enum
from pydantic import BaseModel, Field

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ReviewAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    key_points: list[str]
    summary: str

ai = LazyAgent("anthropic")
analysis = ai.json(
    "Analyze this review: 'The food was amazing but the service was slow'",
    ReviewAnalysis,
)
print(analysis.sentiment)     # Sentiment.NEUTRAL
print(analysis.confidence)    # 0.75
print(analysis.key_points)    # ["Great food quality", "Slow service"]
```

## Agent-level schema

Set a default schema on the agent — every call returns that type:

```python
class QA(BaseModel):
    question: str
    answer: str
    confidence: float

ai = LazyAgent("anthropic", output_schema=QA)
result = ai.chat("What is the speed of light?")

# result.parsed is the QA instance
qa = result.parsed
print(qa.answer)       # "Approximately 299,792,458 meters per second"
print(qa.confidence)   # 0.99

# agent.result returns the typed object when available
print(ai.result.answer)  # same thing
```

## Error handling

Two error types when structured output fails:

```python
from lazybridge.core.types import (
    StructuredOutputParseError,       # JSON was invalid
    StructuredOutputValidationError,  # JSON valid but doesn't match schema
)
from lazybridge import StructuredOutputError  # base class (also re-exported)

try:
    result = ai.json("Generate data", MyModel)
except StructuredOutputParseError as e:
    print(f"Model returned invalid JSON: {e.raw}")
except StructuredOutputValidationError as e:
    print(f"JSON valid but wrong shape: {e}")
except StructuredOutputError as e:
    print(f"Any structured output error: {e}")
```

## Structured output with tools

Combine tools and structured output — the agent uses tools to gather data, then returns a typed result:

```python
class WeatherReport(BaseModel):
    city: str
    temperature: float
    conditions: str
    recommendation: str

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"{city}: 18°C, partly cloudy"

ai = LazyAgent("anthropic")
report = ai.json(
    "Get the weather in London and give me a report",
    WeatherReport,
    tools=[LazyTool.from_function(get_weather)],
)
print(report.recommendation)  # "Bring a light jacket..."
```

## Async version

```python
import asyncio

async def main():
    ai = LazyAgent("anthropic")
    city = await ai.ajson("Tell me about Paris", City)
    print(city.name)

asyncio.run(main())
```

---

## Exercise

1. Create a `BookReview` model with fields: title, author, rating (1-5), pros (list), cons (list)
2. Ask the agent to review a book and return a `BookReview`
3. Try asking for a review of a book that doesn't exist — does the agent hallucinate or refuse?
4. Add a `Field(description=...)` to guide the agent's output format

**Next:** [Module 4: Memory & Conversations](04-memory.md) — build stateful chat conversations.
