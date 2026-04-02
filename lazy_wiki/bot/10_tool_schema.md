# Tool Schema Pipeline — ToolSchemaBuilder, Modes, Type Mapping

This file covers the full pipeline that converts a Python function into a provider-ready `ToolDefinition`.

---

## Import

```python
from lazybridgeframework import (
    ToolSchemaMode,
    ToolSchemaBuilder,
    ToolArgumentValidationError,
    ToolSchemaBuildError,
)
from lazybridgeframework.core.tool_schema import (
    ToolCompileArtifact,
    ToolSourceStatus,
    ArtifactStore,
    InMemoryArtifactStore,
)
```

---

## Overview

When you call `LazyTool.from_function(fn)`, the following pipeline runs automatically:

```
Python function
      │
      ▼
_build_signature_mode()         ← introspect type annotations + docstring
      │
      ▼
ToolDefinition                  ← name, description, parameters (JSON Schema)
      │
      ▼
LazyTool._compiled              ← cached; sent to provider with each request
```

Three modes control how the `ToolDefinition` is built:

| Mode | Schema source | Types | Descriptions | Requires |
|------|--------------|-------|--------------|---------|
| `SIGNATURE` (default) | Type annotations | From hints | From docstring | Nothing |
| `HYBRID` | Type annotations (authoritative) + LLM | From hints | LLM-generated | `schema_llm=` |
| `LLM` | LLM-inferred | LLM-inferred | LLM-generated | `schema_llm=` |

---

## Schema modes in practice

### SIGNATURE (default)

```python
from lazybridgeframework import LazyTool

def search(query: str, max_results: int = 10) -> list[str]:
    """Search the web for results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    ...

tool = LazyTool.from_function(search)
print(tool.definition().parameters)
# {
#   "type": "object",
#   "properties": {
#     "query":       {"type": "string",  "description": "The search query string."},
#     "max_results": {"type": "integer", "description": "Maximum number of results to return."}
#   },
#   "required": ["query"]   ← max_results has default, so not required
# }
```

Descriptions come from:
1. `Annotated[T, "description"]` annotation (highest priority)
2. Google-style or Sphinx-style docstring parameter block
3. Nothing (property present, description absent)

### HYBRID — types from code, descriptions from LLM

Use when your function has good type hints but poor/absent docstrings and you want the LLM to generate descriptions:

```python
llm = LazyAgent("anthropic")

tool = LazyTool.from_function(
    search,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=llm,
)
```

The LLM receives the function source and generates descriptions only. Types always come from the signature — no hallucination risk on types.

Falls back silently to SIGNATURE if `schema_llm` fails or is None (emits a `UserWarning`).

### LLM — full schema from LLM

Use for functions without type hints (e.g. legacy code):

```python
def legacy_api(endpoint, payload, timeout):
    # no type annotations, no docstring
    ...

tool = LazyTool.from_function(
    legacy_api,
    schema_mode=ToolSchemaMode.LLM,
    schema_llm=LazyAgent("anthropic"),
)
```

The LLM infers the full schema from the function's source code and name. Falls back to SIGNATURE if the LLM call fails.

---

## Type annotation → JSON Schema mapping

Full table of how Python annotations are converted:

| Python annotation | JSON Schema |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list` | `{"type": "array"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `list[int]` | `{"type": "array", "items": {"type": "integer"}}` |
| `set[str]` / `frozenset[str]` | `{"type": "array", "uniqueItems": true, "items": {"type": "string"}}` |
| `tuple[str, ...]` | `{"type": "array", "items": {"type": "string"}}` |
| `tuple[str, int, bool]` | `{"type": "array", "prefixItems": [...], "items": false}` |
| `dict` / `dict[str, Any]` | `{"type": "object"}` |
| `Optional[str]` | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` |
| `str \| None` | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` |
| `str \| int` | `{"anyOf": [{"type": "string"}, {"type": "integer"}]}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |
| `MyEnum` (subclass of `Enum`) | `{"enum": [e.value for e in MyEnum]}` |
| `MyModel` (Pydantic `BaseModel`) | Full JSON Schema from `model_json_schema()` |
| `Annotated[str, "desc"]` | `{"type": "string", "description": "desc"}` |
| `Any` / unknown | `{}` (no constraints) |

### Annotated descriptions (recommended pattern)

```python
from typing import Annotated

def search(
    query: Annotated[str, "The search query. Be specific."],
    language: Annotated[str, "ISO 639-1 language code, e.g. 'en', 'it'"] = "en",
    max_results: Annotated[int, "Maximum results to return (1-50)"] = 10,
) -> list[str]:
    ...
```

`Annotated` descriptions take precedence over docstring descriptions.

### Pydantic models as arguments

```python
from pydantic import BaseModel

class SearchOptions(BaseModel):
    query: str
    language: str = "en"
    max_results: int = 10
    safe_search: bool = True

def advanced_search(options: SearchOptions) -> list[str]:
    """Run an advanced search with full options."""
    ...

tool = LazyTool.from_function(advanced_search)
# The schema for `options` uses SearchOptions.model_json_schema() directly
```

### Enum values as constraints

```python
from enum import Enum

class OutputFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "markdown"
    PLAIN = "plain"

def export(content: str, fmt: OutputFormat = OutputFormat.JSON) -> str:
    ...

tool = LazyTool.from_function(export)
# fmt parameter: {"enum": ["json", "markdown", "plain"]}
```

---

## Docstring parameter extraction

SIGNATURE mode extracts parameter descriptions from docstrings automatically.

**Google / NumPy style (recommended):**

```python
def search(query: str, max_results: int = 10) -> list[str]:
    """Search the web.

    Args:
        query: The search query. Be specific and use keywords.
        max_results: Maximum number of results (1-50).
    """
    ...
```

**Sphinx / reST style:**

```python
def search(query: str, max_results: int = 10) -> list[str]:
    """Search the web.

    :param query: The search query. Be specific and use keywords.
    :param int max_results: Maximum number of results (1-50).
    """
    ...
```

Both styles are parsed automatically. `Annotated` descriptions take precedence if both are present.

---

## Argument validation

Before your function is called, arguments from the LLM are validated via a Pydantic model generated from the function's signature. This catches:
- Missing required arguments
- Wrong types (e.g. LLM sends `"10"` instead of `10` — coerced to `int` automatically)
- Extra unexpected arguments (raises if function has no `**kwargs`)

```python
from lazybridgeframework import ToolArgumentValidationError

def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b

tool = LazyTool.from_function(divide)

try:
    tool.run({"a": "not-a-number", "b": 2})
except ToolArgumentValidationError as e:
    print(e)  # "Invalid arguments for 'divide': a: Input should be a valid integer"
```

Coercion example:
```python
# LLM sends string "5" for an int parameter → automatically coerced to 5
tool.run({"a": "5", "b": "2"})   # works, a=5, b=2 after coercion
```

---

## ToolSchemaBuilder — advanced usage

`ToolSchemaBuilder` is the class that runs the pipeline. `LazyTool.from_function` uses a default shared instance.

Use it directly when you need:
- A persistent artifact cache across multiple tools
- `flatten_refs=True` for providers that don't support JSON Schema `$ref`
- Auditing which parameters were LLM-enriched

```python
from lazybridgeframework import ToolSchemaBuilder
from lazybridgeframework.core.tool_schema import InMemoryArtifactStore

# Build with artifact caching
store   = InMemoryArtifactStore()
builder = ToolSchemaBuilder(artifact_store=store)

tool = LazyTool.from_function(
    search,
    schema_builder=builder,       # use this builder instead of the default
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=LazyAgent("anthropic"),
)
```

### flatten_refs — for providers without $ref support

By default, Pydantic model schemas may contain `$ref` / `$defs` entries (JSON Schema style). Anthropic and OpenAI handle these natively. If you use a provider that doesn't:

```python
builder = ToolSchemaBuilder(flatten_refs=True)

tool = LazyTool.from_function(my_fn_with_pydantic_args, schema_builder=builder)
# All $ref references are inlined; no $defs in the output
```

---

## ToolCompileArtifact — auditing schema generation

`build_artifact()` returns a rich object instead of just the definition:

```python
from lazybridgeframework import ToolSchemaBuilder
from lazybridgeframework.core.tool_schema import ToolSchemaMode

builder  = ToolSchemaBuilder()
artifact = builder.build_artifact(
    search,
    mode=ToolSchemaMode.HYBRID,
    schema_llm=LazyAgent("anthropic"),
)

print(artifact.fingerprint)          # "a3f2c1..." — 24-char hex hash of all inputs
print(artifact.source_status)        # "llm_enriched" or "fallback_to_baseline"
print(artifact.llm_enriched_fields)  # frozenset of param names enriched by LLM
print(artifact.warnings)             # tuple of warning strings
print(artifact.cache_hit)            # True if retrieved from ArtifactStore
print(artifact.definition)           # ToolDefinition ready for the provider

# Compare LLM result against the SIGNATURE baseline (drift detection):
if artifact.baseline_definition:
    baseline_params = set(artifact.baseline_definition.parameters["properties"].keys())
    llm_params      = set(artifact.definition.parameters["properties"].keys())
    if baseline_params != llm_params:
        print(f"Drift: LLM added {llm_params - baseline_params}, missed {baseline_params - llm_params}")
```

### ToolSourceStatus values

| Value | Meaning |
|---|---|
| `BASELINE_ONLY` | SIGNATURE mode; no LLM involved |
| `LLM_ENRICHED` | HYBRID mode succeeded; types from signature, descriptions from LLM |
| `LLM_INFERRED` | LLM mode succeeded; full schema from LLM |
| `FALLBACK_TO_BASELINE` | LLM/HYBRID requested but failed or `schema_llm` was None; used SIGNATURE |

---

## Artifact caching

Caching avoids redundant introspection or LLM calls when the same function is used across multiple tool instances:

```python
store   = InMemoryArtifactStore()
builder = ToolSchemaBuilder(artifact_store=store)

tool_a = LazyTool.from_function(search, schema_builder=builder)
tool_b = LazyTool.from_function(search, schema_builder=builder)

tool_a.definition()   # compiles and caches
tool_b.definition()   # cache hit — no recomputation
print(len(store))     # 1 entry
```

The cache key is a 24-char SHA-256 fingerprint of:
- Function qualified name + source code hash
- Effective name and description
- Mode, strict flag, schema_llm identity
- Compiler and prompt template versions

If any of these change (including the function's source code), the fingerprint changes and the cache misses.

Implement `ArtifactStore` for a persistent (e.g. SQLite-backed) cache:

```python
from lazybridgeframework.core.tool_schema import ArtifactStore, ToolCompileArtifact

class MyPersistentStore:
    def get(self, fingerprint: str) -> ToolCompileArtifact | None:
        # load from database
        ...
    def put(self, artifact: ToolCompileArtifact) -> None:
        # save to database
        ...

builder = ToolSchemaBuilder(artifact_store=MyPersistentStore())
```

---

## `tool.compile()` — pre-freeze schema

Force schema compilation before the tool is first used (e.g. at startup to catch errors early):

```python
tool = LazyTool.from_function(search).compile()
# Schema is now frozen in tool._compiled; no recompilation on first use
```

Useful when `schema_mode=LLM` or `HYBRID` and you want LLM calls to happen at init time, not during the first loop step.

```python
tool = LazyTool.from_function(
    search,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=LazyAgent("anthropic"),
).compile()   # LLM call happens here, not at loop time
```
