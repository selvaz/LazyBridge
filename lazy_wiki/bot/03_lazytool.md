# LazyTool — Complete Reference

## 1. Overview

A `LazyTool` wraps either (A) a Python callable or (B) a `LazyAgent`. It always has:
- `name: str` and `description: str`
- a compiled `ToolDefinition` (JSON Schema) — built lazily and cached
- optional `guidance: str` — injected into the calling agent's system prompt
- `run()` / `arun()` — synchronous and async execution

Do not construct `LazyTool` directly. Use the factory methods.

---

## 2. `from_function()` — wrapping a Python callable

```python
@classmethod
def from_function(
    cls,
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    guidance: str | None = None,
    schema_mode: ToolSchemaMode = ToolSchemaMode.SIGNATURE,
    strict: bool = False,
    schema_builder: ToolSchemaBuilder | None = None,
    schema_llm: Any | None = None,
) -> LazyTool:
```

`name` defaults to `func.__name__`. `description` defaults to the first line of `func`'s docstring (or `func.__name__` if there is no docstring).

```python
from lazybridge import LazyTool

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for a query. Returns top results as text."""
    ...

# Schema auto-generated from type hints
tool = LazyTool.from_function(search_web)
# name        = "search_web"
# description = "Search the web for a query."
# schema      = {"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query", "max_results"]}

# Override name/description
tool = LazyTool.from_function(
    search_web,
    name="web_search",
    description="Search the internet for up-to-date information",
    guidance="Use this for any question requiring current data or facts.",
)
```

### Type hint → JSON Schema mapping (SIGNATURE mode)

| Python type | JSON Schema type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| anything else | `"string"` |
| `dict` value in params | used as-is (raw JSON Schema fragment) |

All parameters with type hints appear in `required`. Parameters with no hint become `"string"`.

---

## 3. `from_agent()` — wrapping a LazyAgent

```python
@classmethod
def from_agent(
    cls,
    agent: LazyAgent,
    *,
    name: str | None = None,
    description: str | None = None,
    guidance: str | None = None,
    output_schema: type | dict | None = None,
    native_tools: list | None = None,
    system_prompt: str | None = None,
    strict: bool = False,
) -> LazyTool:
```

Schema is always `{"task": str}`. The orchestrator passes a task string; the delegate's `loop()` or `chat()` receives it.

```python
from lazybridge import LazyAgent, LazyTool
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    findings: list[str]

analyst = LazyAgent("anthropic", name="analyst")

analyst_tool = LazyTool.from_agent(
    analyst,
    name="data_analyst",
    description="Analyse data and return structured insights",
    guidance="Pass raw data in the task. Always return findings as bullet points.",
    output_schema=AnalysisResult,   # resp.parsed is returned instead of resp.content
)
```

### Dispatch logic when the tool is called

1. If `output_schema` is set: `agent.chat(task, output_schema=...)` is called; `resp.parsed` is returned (falls back to `resp.content` if `parsed` is `None`).
2. If the agent has bound tools (`agent.tools`) or `native_tools` is set: `agent.loop(task, tools=agent.tools, native_tools=...)` is called.
3. Otherwise: `agent.chat(task)` is called; `resp.content` is returned.

---

## 4. Schema modes

```python
from lazybridge.lazy_tool import ToolSchemaMode

ToolSchemaMode.SIGNATURE   # default — introspect type hints at tool creation time
ToolSchemaMode.LLM         # an LLM generates the full schema; requires schema_llm=
ToolSchemaMode.HYBRID      # type hints for types, LLM fills in descriptions
```

```python
from lazybridge import LazyAgent, LazyTool
from lazybridge.lazy_tool import ToolSchemaMode

schema_agent = LazyAgent("anthropic")

tool = LazyTool.from_function(
    search_web,
    schema_mode=ToolSchemaMode.LLM,
    schema_llm=schema_agent,        # any LazyAgent used to generate the schema
)
```

Schema is built lazily on first call to `tool.definition()` and cached in `tool._compiled`.

---

## 5. `run()` / `arun()`

```python
def run(self, arguments: dict[str, Any], *, parent: Any = None) -> Any:
async def arun(self, arguments: dict[str, Any], *, parent: Any = None) -> Any:
```

Called by `LazyAgent` internally during `loop()` — you rarely call these directly. `parent` is the calling agent (passed automatically by `loop()`).

Arguments are validated and coerced against the function's type hints before the function is called. `ToolArgumentValidationError` is raised on validation failure.

```python
tool = LazyTool.from_function(search_web)

# Sync
result = tool.run({"query": "AI news", "max_results": 3})

# Async (awaitable — always a coroutine even if the wrapped function is sync)
result = await tool.arun({"query": "AI news"})
```

---

## 6. `guidance` — injected into the calling agent's system prompt

`guidance` is appended to the calling agent's system prompt (not to the tool's JSON Schema). It appears as `[tool_name]\n<guidance text>` in the assembled system string.

```python
from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

tool = LazyTool.from_function(
    get_weather,
    guidance="When the user asks about weather, always ask for the city first if not provided.",
)

ai = LazyAgent("anthropic")
# The guidance text appears in ai's system prompt for any loop() call that includes this tool
result = ai.loop("What's the weather like?", tools=[tool])
```

---

## 7. `specialize()` — create a named variant

Returns a copy of the tool with selective overrides. The cached schema (`_compiled`) is cleared for function-backed tools so it is rebuilt with the new `name`/`description`. Delegate tools keep their fixed `{"task": str}` schema.

```python
from lazybridge import LazyTool

def query_db(region: str, table: str) -> str:
    """Query the database."""
    ...

base_tool = LazyTool.from_function(query_db, description="Query the database")

eu_tool = base_tool.specialize(
    name="query_eu_db",
    description="Query the EU customer database",
    guidance="Filter results to EU region only.",
)

us_tool = base_tool.specialize(
    name="query_us_db",
    description="Query the US customer database",
)
```

`specialize()` accepts: `name`, `description`, `guidance`, `schema_mode`, `strict`. It does not accept `output_schema` or `system_prompt` — those are delegate-only parameters set at `from_agent()` time.

---

## 8. `NormalizedToolSet` — internal normalization

`LazyAgent.loop()` normalizes the combined tool list automatically. Accepts mixed `LazyTool | ToolDefinition | dict`.

```python
from lazybridge import LazyTool
from lazybridge.lazy_tool import NormalizedToolSet
from lazybridge.core.types import ToolDefinition

def search(query: str) -> str:
    """Search."""
    return ""

search_tool = LazyTool.from_function(search)
calc_def    = ToolDefinition("calc", "Calculator", {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]})
lookup_dict = {"name": "lookup", "description": "Look up a value", "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}

tool_set = NormalizedToolSet.from_list([search_tool, calc_def, lookup_dict])

# tool_set.bridges     = [search_tool]              # LazyTool items with callables
# tool_set.definitions = [search_def, calc_def, lookup_def]  # all items as ToolDefinition
# tool_set.registry    = {"search": search_tool}    # name → LazyTool, O(1) lookup
```

`ToolDefinition` and `dict` items have no callable — they rely on the `tool_runner` fallback in `loop()` or native provider handling. Duplicate tool names raise `ValueError`.
