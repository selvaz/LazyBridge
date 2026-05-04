# SupervisorEngine (ext.hil)

**Use `SupervisorEngine`** for full human-in-the-loop control: a REPL
where the operator can call tools, retry agents with feedback, store
keys, or hand control back with `continue`.  It implements the same
`Engine` protocol as `LLMEngine`, so `Agent(engine=SupervisorEngine())`
slots into any pipeline.

**Use `HumanEngine` instead** for approval-only flows where the human
types one response and the pipeline moves on.

## Example

```python
from lazybridge import Agent, Tool, Store
from lazybridge.ext.hil import SupervisorEngine

def search(query: str) -> str:
    """Search the web for query."""
    return f"hits for {query}"

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])
store = Store()
store.write("policy", "publish only peer-reviewed sources")

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[search],
        agents=[researcher],
        store=store,
    ),
    name="supervisor",
)

writer = Agent("claude-opus-4-7", name="writer")

# Pipeline: researcher drafts → supervisor inspects / revises → writer finalises.
agents = [researcher, supervisor, writer]
pipeline = Agent.chain(*agents)
pipeline("AI policy brief")
```

## Pitfalls

- ``input_fn`` is called from a worker thread. If it accesses
  thread-unsafe state (like ``readline`` history), guard it.
- ``agents=`` expects v1 ``Agent`` instances. Duck-typed objects work
  if they expose ``__call__`` / ``run`` and a ``name`` attribute.
- The REPL blocks the human user — if ``timeout=None`` (the default),
  an unattended pipeline hangs forever. Set ``timeout=``+``default=``
  for unattended runs.
- Tool calls in the REPL go via ``run_sync``. If a tool's ``func`` is
  async, it's driven to completion automatically (post-v1 fix).

!!! note "API reference"

    SupervisorEngine(
        *,
        tools: list[Tool | Callable | Agent] = None,
        agents: list[Agent] = None,         # agents the human can retry
        store: Store | None = None,
        input_fn: Callable[[str], str] | None = None,
        ainput_fn: Callable[[str], Awaitable[str]] | None = None,
        timeout: float | None = None,
        default: str | None = None,
    ) -> Engine
    
    Usage: Agent(engine=SupervisorEngine(tools=[...], agents=[researcher]))
    
    REPL commands:
      continue [optional text]        accept; return to caller
      retry <agent>: <feedback>       re-run a registered agent with feedback
      store <key>                     print store[key]
      <tool>(<args>)                  invoke a registered tool

!!! warning "Rules & invariants"

    - ``tools=`` accepts functions, Tool instances, and Agent instances
      uniformly (wrap_tool is applied at __init__). Same contract as
      ``Agent(tools=...)``.
    - The REPL runs on a worker thread so the caller's event loop is not
      blocked. ``input_fn`` is called there; use scripted inputs in tests.
    - ``retry <agent>: <feedback>`` re-runs the named agent with the
      feedback appended to the task. The output replaces the current
      supervisor buffer.
    - Unknown commands print help and re-prompt. ``continue`` is the only
      terminator.
    - Session propagation: an Agent wrapping a SupervisorEngine receives
      session events for AGENT_START / AGENT_FINISH like any other engine.

## See also

- [HumanEngine](human-engine.md) — lighter approval-only variant.
- [Plan](plan.md) — typical container for a supervisor mid-pipeline.
