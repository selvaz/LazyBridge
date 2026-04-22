# Agent.as_tool

`as_tool` is the hinge that makes the "everything is a Tool" contract
real. An `Agent` becomes a `Tool` the moment another `Agent` uses it:
same schema, same calling convention, same observability. The outer
engine does not know or care whether the thing it's calling is a pure
function, an Agent, or an Agent of Agents.

The second job of `as_tool` is to hold a verifier. When you want
*every call* of a sub-agent to be vetted by a judge before returning,
pass `verify=judge_agent`. The judge sees the output and must respond
with `"approved"` to accept; anything else is treated as a rejection
with feedback, and the wrapped agent retries with that feedback injected
into the task. This is the tool-level placement of the verify pattern
— different from `Agent(verify=...)` which gates the agent's own
final output.

## Example

```python
from lazybridge import Agent, LLMEngine

# name="researcher" becomes:
#   - the default Tool.name when this Agent is wrapped via as_tool()
#   - the label in Session.graph / event logs / usage_summary()
researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
# ``system=`` lives on the engine, not on Agent.
judge = Agent(
    engine=LLMEngine("claude-opus-4-7",
                     system='Respond "approved" or "rejected: <reason>".'),
    name="judge",
)

# Implicit: pass the agent, LazyBridge wraps it.
# Tool schema is (task: str) -> str; tool name defaults to researcher.name.
orchestrator = Agent("claude-opus-4-7",
                     tools=[researcher])   # equivalent to researcher.as_tool()

# Explicit + verified: the judge gates every research call.
#   name=         overrides the tool name the orchestrator's LLM sees.
#   description=  is the natural-language tool description handed to the
#                 LLM so it knows when to call this tool.
#   verify=       every call through this tool is judged by ``judge``.
#   max_verify=2  up to 2 retries per call; final attempt returned as-is.
orchestrator = Agent("claude-opus-4-7",
                     tools=[researcher.as_tool(
                         name="research",
                         description="Find 3 high-quality sources for a topic.",
                         verify=judge, max_verify=2,
                     )])
```

## Pitfalls

- A misplaced ``verify=`` can cause a feedback loop if the judge is too
  strict; ``max_verify=2`` is a good default ceiling.
- Long nested chains (``Agent → Agent → Agent``) should share one
  ``Session`` — pass ``session=sess`` on the outer agent only and the
  inner ones will inherit it, so ``usage_summary()`` aggregates
  everything into one view.
- ``as_tool()``'s default schema is ``(task: str) -> str`` regardless of
  the wrapped agent's ``output=``. If you need a typed payload in the
  caller, orchestrate via ``Plan`` with ``Step(output=Model)`` instead.

!!! note "API reference"

    agent.as_tool(
        name: str | None = None,
        description: str | None = None,
        *,
        verify: Agent | Callable[[str], Any] | None = None,
        max_verify: int = 3,
    ) -> Tool
    
    # Tool schema: (task: str) -> str
    
    Usage: Agent("model", tools=[researcher.as_tool()])
           Agent("model", tools=[researcher])   # implicit — wrap_tool auto-calls as_tool()

!!! warning "Rules & invariants"

    - ``as_tool()`` with no arguments produces a Tool named after the agent.
      Passing ``name`` / ``description`` overrides them.
    - ``verify=`` turns the tool into a judge-gated call: every invocation
      runs up to ``max_verify`` times against the judge, retrying with the
      judge's feedback injected into the task. This is the "Option B"
      placement — the judge sits at the tool-call boundary.
    - Passing an ``Agent`` directly to ``tools=[...]`` is equivalent to
      passing ``agent.as_tool()``.
    - Nested agents inherit the outer session and register an ``as_tool``
      edge in the graph automatically (see Agent docs).

## See also

[agent](agent.md), [tool](tool.md), [verify](verify.md),
[session](session.md)
