## signature
# Three ways to turn a Python function into an LLM-callable Tool.

Tool(func, *, mode: Literal["signature", "llm", "hybrid"] = "signature",
     schema_llm: Any | None = None, strict: bool = False)

# Mode recap:
#   "signature" — parse type hints + docstring (default). No LLM cost.
#   "llm"       — call an LLM to infer schema from the function body
#                 and docstring.  Needs schema_llm= (an Agent).
#   "hybrid"    — signature first; LLM fills gaps for missing hints.

# Convenience APIs (no explicit Tool() call needed):
wrap_tool(func_or_agent) -> Tool          # uniform wrapper
build_tool_map(list_of_things) -> dict    # batch wrapping
Agent(..., tools=[func])                  # wrap_tool applied automatically

## rules
- ``mode="signature"`` is the default and produces a schema from type
  hints + docstring (parameter types, return type, description, tool
  name). No LLM is called. Fast, deterministic, free.
- ``mode="llm"`` calls ``schema_llm`` (a cheap Agent) to synthesise a
  JSON schema from the function source + docstring. Pays in tokens but
  works for functions with incomplete or missing hints.
- ``mode="hybrid"`` starts with ``"signature"`` and falls back to
  ``"llm"`` only for parameters lacking hints. Best of both when your
  codebase is mixed.
- The schema is cached per ``Tool`` instance (first ``.definition()``
  call computes it; subsequent calls reuse).
- ``strict=True`` asks the provider to enforce the schema exactly (no
  extra fields, no coercion). Available on Anthropic + OpenAI strict
  modes; increases reliability at the cost of some flexibility.

## narrative
LazyBridge's zero-boilerplate promise rests on `ToolSchemaBuilder`:
any Python callable becomes an LLM-usable tool, no JSON schema written
by hand, no decorator dance. Pass a function to `tools=[...]` and
you're done.

Three modes, because three common situations:

* **Signature mode** — your function has complete type hints and a
  good docstring. The builder extracts everything it needs from
  `inspect.signature` + `typing.get_type_hints` + the docstring. This
  is the default and should be your default choice. No LLM is involved;
  conversion is microseconds.

* **LLM mode** — the function has no hints, or uses exotic types the
  JSON schema world doesn't speak (e.g. NumPy arrays, custom classes).
  The builder sends the function's source and docstring to a cheap LLM
  and asks it to produce a schema. You pay tokens once at construction
  time; subsequent calls use the cached schema.

* **Hybrid mode** — you have a mix. The builder tries signature mode
  first and falls back to LLM for parameters it can't resolve.
  Pragmatic for gradual-typed codebases.

In 95% of cases you don't touch this knob at all — the default
signature mode just works. Reach for `"llm"` when you're wrapping a
legacy function you don't want to annotate, or a function whose
argument types genuinely need explanation beyond a type name (e.g.
natural-language parameters like "SQL query dialect").

## example
```python
from lazybridge import Agent, Tool

# --- Signature mode (default, no LLM) -----------------------------
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression and return the result.

    Supports +, -, *, /, parentheses.
    """
    return eval(expression)

Agent("claude-opus-4-7", tools=[calculate])   # schema auto-inferred

# --- LLM mode (schema synthesised by a cheap Agent) ---------------
from lazybridge import Agent

tiny = Agent.from_provider("anthropic", tier="cheap", name="schema_bot")

def legacy_func(data, opts=None):
    """Transform the incoming payload per options. data is a dict of
    readings {timestamp: value}; opts controls resampling.
    """
    ...

legacy_tool = Tool(legacy_func, mode="llm", schema_llm=tiny)
Agent("claude-opus-4-7", tools=[legacy_tool])

# --- Hybrid (signature where possible, LLM where missing) ---------
def partial_hint(query: str, opts=None) -> list:
    """Search and return matches. opts is a dict of filters."""
    ...

Agent("claude-opus-4-7",
      tools=[Tool(partial_hint, mode="hybrid", schema_llm=tiny)])

# --- wrap_tool: uniform conversion -------------------------------
from lazybridge.tools import wrap_tool, build_tool_map

tool_1 = wrap_tool(calculate)                  # function → Tool
tool_2 = wrap_tool(legacy_tool)                 # Tool → Tool (idempotent)
tool_3 = wrap_tool(Agent("claude-opus-4-7"))    # Agent → Tool (via as_tool)

tools_by_name = build_tool_map([calculate, tool_2, Agent(...)])
```

## pitfalls
- ``mode="llm"`` without ``schema_llm=`` silently falls back to
  ``"signature"`` (with warnings). Always pass the schema_llm if you
  pick LLM / hybrid mode.
- Calling ``Tool(func).definition()`` forces the schema computation.
  If ``mode="llm"``, this triggers an LLM call at construction time —
  don't build tools on import if you're latency-sensitive.
- ``strict=True`` is opinionated about JSON schema shape. Tools that
  rely on extra kwargs or variadic args may fail strict validation;
  try without strict first.

## see-also
[tool](tool.md), [agent](agent.md), [native_tools](native-tools.md)
