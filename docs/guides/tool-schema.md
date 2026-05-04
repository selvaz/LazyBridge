# Function → Tool (schema modes)

**Use `mode="signature"`** (the default) for any function with type
hints and a docstring.  The schema is produced deterministically with no
LLM cost — the right choice for >95% of user code.

**Use `mode="llm"`** for legacy functions you can't easily annotate —
opaque `**kwargs`, third-party callables, auto-generated wrappers.  Pay
tokens once at construction time.

**Use `mode="hybrid"`** when the codebase is mixed: the signature path
covers what it can, the LLM only fills annotated gaps.

## Example

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

## Pitfalls

- ``mode="llm"`` without ``schema_llm=`` silently falls back to
  ``"signature"`` (with warnings). Always pass the schema_llm if you
  pick LLM / hybrid mode.
- Calling ``Tool(func).definition()`` forces the schema computation.
  If ``mode="llm"``, this triggers an LLM call at construction time —
  don't build tools on import if you're latency-sensitive.
- ``strict=True`` is opinionated about JSON schema shape. Tools that
  rely on extra kwargs or variadic args may fail strict validation;
  try without strict first.

!!! note "API reference"

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

!!! warning "Rules & invariants"

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

## See also

- [Tool](tool.md) — the wrapper that consumes a schema mode.
- [Native tools](native-tools.md) — the no-schema-needed alternative.
