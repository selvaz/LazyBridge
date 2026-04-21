## signature
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

## rules
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

## narrative
`SupervisorEngine` is a full human-in-the-loop REPL. The human sees
the incoming task and can continue, run tools, invoke a specific
agent with feedback, or inspect the shared store — before finally
returning output to the rest of the pipeline.

Use it when the human is an **operator** (not just an approver):
debugging pipelines, running pilot deployments, dev-loop investigations.
For lighter roles (sign off a draft, fill a form) use `HumanEngine`.

Because it implements the same `Engine` protocol as `LLMEngine`, an
`Agent(engine=SupervisorEngine(...))` plugs into `Agent.chain`,
`Agent.parallel`, or `Plan` the same way any other agent does.

## example
```python
from lazybridge import Agent, SupervisorEngine, Tool, Store

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
pipeline = Agent.chain(researcher, supervisor, writer)
pipeline("AI policy brief")
```

## pitfalls
- ``input_fn`` is called from a worker thread. If it accesses
  thread-unsafe state (like ``readline`` history), guard it.
- ``agents=`` expects v1 ``Agent`` instances. Duck-typed objects work
  if they expose ``__call__`` / ``run`` and a ``name`` attribute.
- The REPL blocks the human user — if ``timeout=None`` (the default),
  an unattended pipeline hangs forever. Set ``timeout=``+``default=``
  for unattended runs.
- Tool calls in the REPL go via ``run_sync``. If a tool's ``func`` is
  async, it's driven to completion automatically (post-v1 fix).

## see-also
[human_engine](human-engine.md), [agent](agent.md),
[plan](plan.md),
decision tree: [human_engine_vs_supervisor](../decisions/human-engine-vs-supervisor.md)
