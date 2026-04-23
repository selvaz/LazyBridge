## signature
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

## rules
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

## example
```python
from lazybridge import Agent

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
judge      = Agent("claude-opus-4-7", name="judge",
                   system='Respond "approved" or "rejected: <reason>".')

# Implicit: pass the agent, LazyBridge wraps it.
orchestrator = Agent("claude-opus-4-7",
                     tools=[researcher])   # equivalent to researcher.as_tool()

# Explicit + verified: the judge gates every research call.
orchestrator = Agent("claude-opus-4-7",
                     tools=[researcher.as_tool(
                         name="research",
                         description="Find 3 high-quality sources for a topic.",
                         verify=judge, max_verify=2,
                     )])
```

## pitfalls
- A misplaced ``verify=`` can cause a feedback loop if the judge is too
  strict; ``max_verify=2`` is a good default ceiling.
- Long nested chains (``Agent → Agent → Agent``) should share one
  ``Session`` — pass ``session=sess`` on the outer agent only and the
  inner ones will inherit it, so ``usage_summary()`` aggregates
  everything into one view.
- ``as_tool()``'s default schema is ``(task: str) -> str`` regardless of
  the wrapped agent's ``output=``. If you need a typed payload in the
  caller, orchestrate via ``Plan`` with ``Step(output=Model)`` instead.

