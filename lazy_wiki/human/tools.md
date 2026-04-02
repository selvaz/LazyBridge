# LazyTool — Tools and Agent Delegation

Tools let the LLM call your code. `LazyTool` handles schema generation, execution, and optional guidance automatically.

---

## Wrapping a Python function

```python
from lazybridge import LazyTool

def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°{unit[0].upper()}, sunny"

tool = LazyTool.from_function(get_weather)
```

That's all. LazyTool:
- Uses `get_weather` as the tool name
- Uses the first line of the docstring as the description
- Generates the JSON Schema from the type hints automatically

### Type hint → JSON Schema mapping

LazyBridge reads your type annotations and converts them automatically. Here is the full mapping:

| Python annotation | JSON Schema |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `dict` | `{"type": "object"}` |
| `Optional[str]` / `str \| None` | `{"anyOf": [{"type":"string"}, {"type":"null"}]}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |
| `MyEnum` (Enum subclass) | `{"enum": [e.value for e in MyEnum]}` |
| `MyModel` (Pydantic BaseModel) | Full JSON Schema from Pydantic |
| `Annotated[str, "description"]` | `{"type": "string", "description": "description"}` |

A parameter is **required** in the schema when it has no default value. Parameters with a default (e.g. `unit: str = "celsius"`) are optional.

### Adding descriptions with Annotated

The cleanest way to add per-parameter descriptions — no docstring needed:

```python
from typing import Annotated

def search_web(
    query: Annotated[str, "The search query. Be specific, use keywords."],
    language: Annotated[str, "ISO 639-1 language code, e.g. 'en', 'it'"] = "en",
    max_results: Annotated[int, "Maximum results (1–50)"] = 10,
) -> list[str]:
    ...

tool = LazyTool.from_function(search_web)
# Each parameter's description is included in the tool schema automatically
```

### Adding descriptions via docstring

Google-style and Sphinx-style docstrings are also parsed:

```python
def search_web(query: str, language: str = "en", max_results: int = 10) -> list[str]:
    """Search the web and return a list of result URLs.

    Args:
        query: The search query. Be specific, use keywords.
        language: ISO 639-1 language code, e.g. 'en', 'it'.
        max_results: Maximum results to return (1–50).
    """
    ...
```

If both `Annotated` and docstring descriptions are present, `Annotated` wins.

### Pydantic models as arguments

Pass a Pydantic model as a parameter type and its full schema is included automatically:

```python
from pydantic import BaseModel

class SearchOptions(BaseModel):
    query: str
    language: str = "en"
    max_results: int = 10
    safe_search: bool = True

def advanced_search(options: SearchOptions) -> list[str]:
    """Run an advanced web search."""
    ...

tool = LazyTool.from_function(advanced_search)
# The LLM sees the full SearchOptions schema including all its fields
```

### Enum constraints

```python
from enum import Enum

class Format(str, Enum):
    JSON     = "json"
    MARKDOWN = "markdown"
    PLAIN    = "plain"

def export(content: str, fmt: Format = Format.JSON) -> str:
    """Export content in the given format."""
    ...

tool = LazyTool.from_function(export)
# fmt schema: {"enum": ["json", "markdown", "plain"]}
# The LLM can only pass one of those three strings
```

### Override name and description

```python
tool = LazyTool.from_function(
    get_weather,
    name="weather_checker",
    description="Look up the current weather in any city worldwide.",
)
```

---

## Schema modes — SIGNATURE, HYBRID, LLM

By default LazyBridge uses `SIGNATURE` mode: it reads your type annotations. Two other modes are available when you need better descriptions.

### SIGNATURE (default) — fast, deterministic

```python
from lazybridge import LazyTool, ToolSchemaMode

tool = LazyTool.from_function(search_web)                          # implicit
tool = LazyTool.from_function(search_web, schema_mode=ToolSchemaMode.SIGNATURE)  # explicit
```

Use when: you have good type hints and/or docstrings.

### HYBRID — types from code, descriptions from LLM

```python
from lazybridge import LazyAgent

llm  = LazyAgent("anthropic")
tool = LazyTool.from_function(
    search_web,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=llm,
)
```

The LLM reads your function source and writes the description for each parameter. Types always come from the signature — not from the LLM — so there is no type hallucination risk.

Use when: your function has type hints but no/poor docstrings and you want richer descriptions without writing them manually.

Falls back to SIGNATURE silently if the LLM call fails.

### LLM — full schema from the model

```python
tool = LazyTool.from_function(
    legacy_function,       # no type hints, no docstrings
    schema_mode=ToolSchemaMode.LLM,
    schema_llm=llm,
)
```

The LLM infers the entire schema (name, description, parameter names, types, required flags) from the function source code. Use for legacy code without annotations.

Falls back to SIGNATURE silently if the LLM call fails.

### Pre-compiling the schema

Force schema generation at startup (useful with LLM/HYBRID to avoid schema calls during the first loop step):

```python
tool = LazyTool.from_function(
    search_web,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=llm,
).compile()   # LLM call happens here, not during the first agent loop
```

---

## Argument validation

Before your function is called, arguments from the LLM are automatically validated:

```python
from lazybridge import ToolArgumentValidationError

def divide(a: int, b: int) -> float:
    return a / b

tool = LazyTool.from_function(divide)

# Type coercion: "5" → 5 automatically
tool.run({"a": "5", "b": "2"})   # works fine

# Validation error on truly invalid input
try:
    tool.run({"a": "not-a-number", "b": 2})
except ToolArgumentValidationError as e:
    print(e)
```

---

## Guidance — hints for the calling model

Guidance is injected into the *calling agent's system prompt*. It tells the LLM *how* to use the tool, not what the tool does.

```python
tool = LazyTool.from_function(
    get_weather,
    guidance="Call this before answering any weather question. Always ask for the city if not provided.",
)
```

This text appears in the orchestrator's system prompt, not in the tool schema. Use it to guide the LLM's decision to call the tool.

---

## Using tools in a loop

```python
from lazybridge import LazyAgent

ai = LazyAgent("anthropic")
result = ai.loop(
    "What's the weather in Rome, Paris, and Tokyo?",
    tools=[tool],
)
print(result.content)
```

`loop()` runs until the model stops requesting tools (or hits `max_steps=8` by default).

---

## Delegating to another agent

Wrap a `LazyAgent` as a tool so an orchestrator can call it:

```python
researcher = LazyAgent(
    "anthropic",
    name="researcher",
    description="Researches any topic and returns a detailed summary.",
)

# Option A: from the agent
research_tool = researcher.as_tool()

# Option B: explicit factory
research_tool = LazyTool.from_agent(
    researcher,
    guidance="Use this for any question that requires current or detailed information.",
)

orchestrator = LazyAgent("anthropic")
result = orchestrator.loop(
    "Prepare a brief on open-source AI models released this year",
    tools=[research_tool],
)
```

The schema is always `{"task": str}`. The orchestrator passes a task string; the researcher runs `loop(task)` and returns its output.

---

## Using a session's pipeline as a tool

```python
from lazybridge import LazySession

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)
analyst    = LazyAgent("openai",    name="analyst",    session=sess)

pipeline_tool = sess.as_tool(
    "research_and_analyse",
    "Runs research and analysis pipeline, returns final report",
    entry_agent=researcher,
)

master = LazyAgent("anthropic")
master.loop("Investigate three topics for our quarterly report", tools=[pipeline_tool])
```

---

## Tool variants with specialize()

Create named versions of the same tool with different behaviour:

```python
base = LazyTool.from_function(search_web, description="Search the web")

eu_search = base.specialize(
    name="search_eu",
    description="Search European news sources",
    guidance="Always filter to European sources published in the last 48 hours.",
)

us_search = base.specialize(
    name="search_us",
    description="Search US news sources",
    guidance="Always filter to US sources published in the last 48 hours.",
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop(
    "Compare today's AI news coverage in the EU vs US",
    tools=[eu_search, us_search],
)
```

---

## Native provider tools

Some LLM providers expose **built-in server-side tools** that run on their infrastructure — no Python function needed. Examples: Anthropic's web search, OpenAI's code interpreter, Google's search grounding.

Use `NativeTool` to activate them:

```python
from lazybridge import LazyAgent
from lazybridge.core.types import NativeTool

ai = LazyAgent("anthropic")
resp = ai.chat(
    "What are the top AI research papers published this week?",
    native_tools=[NativeTool.WEB_SEARCH],
)
print(resp.content)

# You can also pass the string directly
resp = ai.chat("Latest news on fusion energy", native_tools=["web_search"])
```

### Available native tools

| `NativeTool` | Anthropic | OpenAI | Google |
|---|:---:|:---:|:---:|
| `WEB_SEARCH` | ✓ | ✓ | ✓ |
| `CODE_EXECUTION` | ✓ | ✓ | — |
| `FILE_SEARCH` | — | ✓ | — |
| `COMPUTER_USE` | ✓ | ✓ | — |
| `GOOGLE_SEARCH` | — | — | ✓ |
| `GOOGLE_MAPS` | — | — | ✓ |

Unsupported tools for a given provider are silently ignored with a warning — the call never fails.

### Reading grounding sources

When a web search native tool is used, the provider returns citations alongside the answer:

```python
resp = ai.chat("Who won the last Formula 1 championship?", native_tools=[NativeTool.WEB_SEARCH])

for src in resp.grounding_sources:
    print(src.url)
    print(src.title)
    print(src.snippet)

# Google also exposes the actual queries issued
print(resp.web_search_queries)   # e.g. ["F1 2025 champion", "Formula 1 championship winner"]
```

### Native tools in a loop

Native tools work with `loop()` too — useful when the model needs to search multiple times:

```python
result = ai.loop(
    "Research the pros and cons of three different EV battery technologies",
    native_tools=[NativeTool.WEB_SEARCH],
    max_steps=10,
)
print(result.content)
```

You can combine native tools with your own `LazyTool` functions in the same call:

```python
result = ai.loop(
    "Search for today's news on quantum computing, then format it as a structured report",
    tools=[format_report_tool],
    native_tools=[NativeTool.WEB_SEARCH],
)
```

### How it works under the hood

- **Anthropic**: native tools are passed as typed blocks (`"type": "web_search_20260209"`); the required beta headers are added automatically.
- **OpenAI**: when native tools are present, LazyBridge automatically routes to the **Responses API** instead of Chat Completions.
- **Google**: native tools are passed as `ToolConfig.GoogleSearchRetrieval` (grounding config), not as function definitions.

You do not need to configure any of this — LazyBridge handles the provider-specific differences transparently.

---

## Mixing tool types

You can mix `LazyTool`, raw `ToolDefinition`, and dicts in the same list:

```python
from lazybridge.core.types import ToolDefinition

raw = ToolDefinition(
    name="legacy_api",
    description="Query our legacy API",
    parameters={"type": "object", "properties": {"endpoint": {"type": "string"}}, "required": ["endpoint"]},
)

result = ai.loop(
    "Query the system",
    tools=[my_lazy_tool, raw],
    tool_runner=lambda name, args: my_legacy_api(args["endpoint"]),  # handles raw tools
)
```

`tool_runner` is a fallback for tools that don't have a LazyTool callable.

---

## Structured output from a delegated agent

If you want the sub-agent to return a typed object:

```python
from pydantic import BaseModel

class ResearchResult(BaseModel):
    topic: str
    summary: str
    sources: list[str]

research_tool = LazyTool.from_agent(
    researcher,
    output_schema=ResearchResult,
)
```

The orchestrator receives a `ResearchResult` instance as the tool result.
