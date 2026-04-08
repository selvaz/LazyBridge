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

| `output_schema` | tools / native_tools | Method called | Returns |
|---|---|---|---|
| set | set | `agent.loop(task, tools=..., output_schema=...)` | `resp.parsed` (fallback: `resp.content`) |
| set | — | `agent.chat(task, output_schema=...)` | `resp.parsed` (fallback: `resp.content`) |
| — | set | `agent.loop(task, tools=..., native_tools=...)` | `resp.content` |
| — | — | `agent.chat(task)` | `resp.content` |

If `output_schema` is set and the result fails schema validation, a `ValueError` is raised with the validation error message.

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

## 8. `save()` / `load()` — tool persistence

Tools can be serialised to plain Python files and reloaded later — across processes, machines, or sessions. The generated file is human-readable, version-control-friendly, and fully editable.

### Why this matters

Without persistence, every tool definition lives only in memory. With `save()`/`load()`:
- **Share tools across projects** without reimporting the original module
- **Version-control tool definitions** — diffs are readable Python, not binary blobs
- **Deploy tools to remote workers** by shipping a single `.py` file
- **Edit the generated file** and reload — the tool updates automatically on next `load()`

### `save(path)`

```python
tool.save("auto_tool/search_web.py")
```

Generates a `.py` file containing:

```python
# LAZYBRIDGE_GENERATED_TOOL v1
# source: /path/to/tools.py::search_web (line 12)

import requests  # ← imports extracted from original module

from lazybridge import LazyTool


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for a query. Returns top results as text."""
    # ... original function body ...


tool = LazyTool.from_function(
    search_web,
    description="Search the web for a query. Returns top results as text.",
    guidance="Use for any question requiring current data or facts.",
)
```

For **agent-backed tools**, the generated file contains a `LazyAgent` constructor and `agent.as_tool()`:

```python
# LAZYBRIDGE_GENERATED_TOOL v1
# NOTE: API keys are not serialized.
# Set the appropriate environment variable before loading
# (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).

from lazybridge import LazyAgent, LazyTool


agent = LazyAgent(
    "anthropic",
    name="analyst",
    model="claude-sonnet-4-6",
    system="You are a senior data analyst.",
)

tool = agent.as_tool(
    name="data_analyst",
    description="Analyse data and return structured insights",
    guidance="Pass raw data in the task.",
)
```

Parent directories are created automatically.

### `load(path)` — classmethod

```python
tool = LazyTool.load("auto_tool/search_web.py")
result = tool.run({"query": "AI news"})
```

Executes the file via `importlib` and returns the `tool` variable. The tool is fully operational immediately after load — `run()`, `arun()`, `definition()`, `specialize()` all work as normal.

### Round-trip example

```python
from lazybridge import LazyAgent, LazyTool

def summarise(text: str, max_words: int = 100) -> str:
    """Summarise text in at most max_words words."""
    return text[:max_words * 5]  # placeholder

# Save
tool = LazyTool.from_function(
    summarise,
    guidance="Use when you need a short version of a long document.",
)
tool.save("auto_tool/summarise.py")

# --- later, in a different script or process ---
tool = LazyTool.load("auto_tool/summarise.py")

agent = LazyAgent("anthropic")
result = agent.loop("Summarise this report: ...", tools=[tool])
```

### Security model and safeguards

`load()` only accepts files that were generated by `save()`. Three layers of protection:

| Safeguard | What it does |
|---|---|
| **Sentinel header** | `load()` reads the first line before executing anything. If `# LAZYBRIDGE_GENERATED_TOOL v1` is absent → `ValueError`. Prevents `load()` from executing arbitrary `.py` files. |
| **Path validation** | Both `save()` and `load()` reject paths containing `..`. Prevents path traversal attacks. |
| **No API key serialization** | Agent-backed tools never include the `api_key` value. A comment reminds you to set the environment variable. |

**Critical rule**: never expose `LazyTool.load` as an agent tool (`LazyTool.from_function(LazyTool.load)`). `load()` executes the target file — a path controlled by an LLM is a code-execution vulnerability.

### Limitations

| Case | Behaviour |
|---|---|
| Lambda (`lambda x: x`) | `ValueError` — lambdas have no recoverable source |
| Function defined in REPL | `ValueError` — `inspect.getsource()` fails |
| Closure capturing external variables | `ValueError` — captured variables are not serialized |
| `from_agent()` with `output_schema=` | Schema class name appears as a comment — re-attach manually after load |

**Workaround for REPL / closure**: define the function in a `.py` file before calling `from_function()`.

---

## 9. `NormalizedToolSet` — internal normalization

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

---

## 5. `parallel()` — fan-out pipeline tool

Runs all participants concurrently on the same task. No `LazySession` required.
Participants are **cloned per invocation** — `participant._last_output` on the
original is `None` after the run. Use the tool's return value.

```python
from lazybridge import LazyAgent, LazyTool

us     = LazyAgent("anthropic", name="us",     system="Report AI news from the US.")
europe = LazyAgent("openai",    name="europe",  system="Report AI news from Europe.")
asia   = LazyAgent("google",    name="asia",    system="Report AI news from Asia.")

news_tool = LazyTool.parallel(
    us, europe, asia,
    name="world_news",
    description="Parallel AI news summary from US, Europe, and Asia",
    combiner="concat",   # default — outputs joined with [agent_name] headers
)

# Pass to an orchestrator agent
orchestrator.loop("Summarise today's AI news", tools=[news_tool])
```

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `*participants` | — | LazyAgent or LazyTool instances |
| `name` | required | Tool name |
| `description` | required | Tool description |
| `combiner` | `"concat"` | `"concat"` joins outputs with agent headers; `"last"` returns only the last |
| `native_tools` | `None` | NativeTool list passed to all agent participants |
| `session` | `None` | Validation-only: raises ValueError if any agent is bound to a conflicting session |
| `guidance` | `None` | Hint injected into the tool description |

---

## 6. `chain()` — sequential pipeline tool

Runs participants in order. Each agent receives the previous step's output
as context (agent→agent) or as its task (tool→agent). No `LazySession` required.
Participants are **cloned per invocation** — `participant._last_output` on the
original is `None` after the run. Use the return value of `tool.run()` or
set `output_schema` on the last step.

```python
from lazybridge import LazyAgent, LazyTool

researcher = LazyAgent("anthropic", name="researcher",
                       system="Find and summarise research on the given topic.")
analyst    = LazyAgent("openai",    name="analyst",
                       system="Analyse the research and draw conclusions.")

pipeline = LazyTool.chain(
    researcher, analyst,
    name="research_pipeline",
    description="Research then analyse — returns analyst's conclusions",
)

orchestrator.loop("Analyse fusion energy breakthroughs", tools=[pipeline])
```

**Handoff semantics:**

| Previous step | Next step receives |
|---|---|
| LazyAgent | Original task + previous agent's output injected as context |
| LazyTool | Tool's output becomes the new task |

**Parameters:** same as `parallel()` minus `combiner`.

---

## 7. `save()` and pipeline tools

`save()` raises `ValueError` on `chain()` / `parallel()` tools — they are
runtime compositions (closures over participant references) and cannot be
serialised to a static file.

```python
pipeline = LazyTool.chain(researcher, analyst, name="p", description="t")
pipeline.save("pipeline.py")
# ValueError: LazyTool 'p' is a chain or parallel pipeline tool and
#   cannot be serialized — it is a runtime composition.
#   Save individual participants via agent.as_tool().save() instead.
```

To persist the pipeline, save each participant's agent tool instead:

```python
researcher.as_tool(name="researcher_tool", description="...").save("researcher.py")
analyst.as_tool(name="analyst_tool", description="...").save("analyst.py")
```

---

## 8. Clone behaviour — `participant._last_output` after run

`LazyTool.parallel()` and `LazyTool.chain()` clone participants per invocation.
The **original** participant's `_last_output` is `None` after the call.

```python
researcher = LazyAgent("anthropic", name="researcher")
pipeline = LazyTool.chain(researcher, analyst, name="p", description="t")

result = pipeline.run({"task": "Analyse X"})

print(researcher._last_output)   # None — clone ran, not the original
print(result)                    # "..." — analyst's conclusion (use this)
```

If you need the intermediate agent's output, use `output_schema` on the chain
step and read the returned Pydantic object, or restructure the pipeline so the
intermediate result is returned directly.
