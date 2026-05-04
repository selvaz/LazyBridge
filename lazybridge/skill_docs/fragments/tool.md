## signature
Tool(
    func: Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    guidance: str | None = None,
    mode: Literal["signature", "llm", "hybrid"] = "signature",
    schema_llm: Any | None = None,
    strict: bool = False,
) -> Tool

Tool.definition() -> ToolDefinition
await Tool.run(**kwargs) -> Any
Tool.run_sync(**kwargs) -> Any   # handles async ``func`` transparently

wrap_tool(obj) -> Tool   # converts functions / Agents / Tools uniformly
build_tool_map(tools: list) -> dict[str, Tool]

## rules
- Schema generation is automatic from ``func``'s type hints and docstring
  in ``mode="signature"`` (default). Use ``mode="llm"`` to let an LLM
  synthesise the schema, ``mode="hybrid"`` for both.
- ``name`` defaults to ``func.__name__``. Names are API-facing; pick
  stable ones.
- ``strict=True`` enables provider-strict JSON-schema validation on tool
  arguments (Anthropic / OpenAI strict mode).
- ``run`` is async; ``run_sync`` auto-detects coroutine functions and
  drives them to completion so REPL callers (e.g. SupervisorEngine) never
  see a raw coroutine.

## narrative
**Use `Tool` for** any callable you want an LLM to invoke — local
Python functions, agents wrapped via `as_tool`, or pre-built
provider-specific helpers. Pass plain functions to `tools=[...]` and
the framework wraps them automatically; reach for `Tool(...)` only
when you need to override the name, description, or schema mode.

**Reach for `mode="llm"` / `"hybrid"`** when the function lacks type
hints and you can't add them — see [Function → Tool](tool-schema.md)
for the trade-offs.

## example
```python
from lazybridge import Tool, Agent

def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)  # noqa: S307  (trusted inputs only)

# Implicit: pass the function, LazyBridge wraps it.
Agent("claude-opus-4-7", tools=[calculate])("what is 17 * 23?")

# Explicit: override the name or strictness.
calc_tool = Tool(calculate, name="calc", strict=True,
                 description="Evaluate an arithmetic expression.")
Agent("claude-opus-4-7", tools=[calc_tool])("...")

# An Agent is also a Tool — no ceremony.
researcher = Agent("claude-opus-4-7", tools=[search], name="researcher")
orchestrator = Agent("claude-opus-4-7", tools=[researcher])
```

## pitfalls
- A function with no type hints produces an empty JSON schema and the
  LLM will not know how to call it. Always annotate parameters.
- A docstring is part of the contract the LLM reads. "Returns the
  weather" is weaker than "Returns the current temperature in Celsius
  and a one-word condition (sunny / cloudy / rainy) for ``city``."
- ``strict=True`` rejects optional / defaulted args under some providers;
  if a call fails with "unknown parameter", try ``strict=False``.

## see-also
- [Function → Tool](tool-schema.md) — schema modes (signature / llm / hybrid).
- [Native tools](native-tools.md) — provider-hosted alternatives.
- [Agent](agent.md) — the surface that consumes tools.
