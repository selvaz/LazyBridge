# SupervisorEngine

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

## REPL commands — what they do and what they record

The REPL accepts four command shapes.  Every command — including the
empty line and unrecognised input — is recorded on the session's
event log as a **`HIL_DECISION` event** so you get a full audit trail
of the human's interaction:

| Command | Effect | `HIL_DECISION.kind` | `result` payload |
|---|---|---|---|
| `continue` or `continue <text>` | Accept current output (or override with `<text>`) and return it to the caller.  This is the only terminator. | `continue` | the final text returned |
| `retry <agent>: <feedback>` | Re-run `<agent>` (which must have been registered via `agents=[...]`) with `<feedback>` appended to its task as `"\n\nFeedback: ..."`.  The new output replaces the current one. | `retry` | new agent output, or `error: ...` |
| `store <key>` | Read `store[<key>]` (requires `store=`).  Read-only — cannot mutate. | `store` | stringified value, or `"no store"` |
| `<tool>(<args>)` | Invoke a registered tool.  Anything between the first `(` and last `)` is passed as the raw argument string to the tool's first required parameter.  Single-token quoted args (`'...'` / `"..."`) are stripped. | `tool` | tool return value (first 500 chars) |
| *(empty line)* | No-op, re-prompt. | `empty` | — |
| *anything else* | Print a help line, re-prompt. | `unknown` | — |

The audit events name the agent (`agent_name`), the literal command
(`command`), and a truncated result.  An exporter on the session
(`JsonFileExporter`, `OTelExporter`) will receive these events just
like any other event type; filter on `EventType.HIL_DECISION` to
build a compliance log.

## Example

```python
# What this shows: a three-agent pipeline where the human operator sits
# in the middle. They see the researcher's draft, can invoke tools,
# retry the researcher with feedback, consult the shared store, and
# only then hand off to the writer.
# Why: the researcher is cheap to rerun, the writer is expensive to
# rerun, and the human is the authority on when the draft is ready.
# SupervisorEngine makes all three capabilities (tool calls, agent
# retry, store lookup) available AT THE SAME REPL without writing any
# orchestration code.

from lazybridge import Agent, SupervisorEngine, Tool, Store

def search(query: str) -> str:
    """Search the web for query."""
    return f"hits for {query}"

researcher = Agent("claude-opus-4-7", name="researcher", tools=[search])

store = Store()
store.write("policy", "publish only peer-reviewed sources")

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[search],          # REPL command: search("AI safety")
        agents=[researcher],     # REPL command: retry researcher: be more specific
        store=store,             # REPL command: store policy
        # timeout / default can be set here too for unattended runs —
        # see the "Unattended fallback" subsection below.
    ),
    name="supervisor",            # matters: appears in HIL_DECISION events
                                  # and SupervisorEngine's header output
)

writer = Agent("claude-opus-4-7", name="writer")

# Pipeline: researcher drafts → supervisor inspects / revises → writer finalises.
# Note: researcher appears BOTH as the first chain step AND as a
# retryable agent inside the supervisor. That's intentional — the
# human can re-drive the same upstream agent with feedback without
# restarting the entire chain.
pipeline = Agent.chain(researcher, supervisor, writer)
pipeline("AI policy brief")
```

After `pipeline(...)` returns, `supervisor`'s `HIL_DECISION` events
tell you exactly which commands the human issued.  All nested agents
(`researcher`, `writer`) inherit the chain's session automatically,
so `session.usage_summary()` aggregates cost across the whole tree
including retries.

## Async REPL for async hosts (`ainput_fn`)

The default REPL runs on a worker thread — fine for a CLI script, but
brittle inside async web frameworks (FastAPI, Starlette) where
cancellation needs to propagate through the event loop.  Passing
`ainput_fn` swaps in an async-native prompt loop.

```python
# What this shows: hooking SupervisorEngine into an asyncio-native
# prompt source (here a queue fed from a WebSocket).  Cancellation on
# the surrounding task now propagates correctly — Ctrl-C or a client
# disconnect tears down the REPL loop promptly instead of leaving a
# daemon thread blocked on input().
# Why: the sync input_fn runs on asyncio.to_thread, so asyncio cancel
# scopes can't interrupt input() once it has started reading.
# ainput_fn avoids that entirely.

import asyncio

async def websocket_input_fn(prompt: str) -> str:
    await send_to_browser(prompt)             # hypothetical WS helper
    return await receive_from_browser()        # hypothetical WS helper

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[search],
        agents=[researcher],
        store=store,
        ainput_fn=websocket_input_fn,   # <- overrides the sync input_fn
    ),
    name="supervisor",
)
```

`input_fn` and `ainput_fn` are mutually exclusive at runtime — if both
are supplied, `ainput_fn` wins and the async path is used.  For tests,
`lazybridge.testing.scripted_ainputs(...)` returns a compatible
coroutine; see the [testing guide](testing.md) for a full harness.

## Unattended fallback (`timeout` + `default`)

A SupervisorEngine with no `timeout` will block forever on the prompt.
In scheduled pipelines or overnight batch jobs you want the loop to
proceed on its own if the operator is unavailable.

```python
supervisor = Agent(
    engine=SupervisorEngine(
        tools=[search], agents=[researcher], store=store,
        timeout=300.0,                 # seconds to wait for each prompt
        default="continue",            # executed as the next "input" on timeout
    ),
    name="supervisor",
)
```

Semantics: the engine starts a watcher thread around `input_fn`; if
the timeout fires and `default` is non-`None`, that string is treated
as the next REPL command (`"continue"` accepts the current output and
exits).  If `default` is `None` at the timeout, the engine raises
`TimeoutError` — fail-loud so you see the stuck pipeline.

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

[human_engine](human-engine.md), [agent](agent.md),
[plan](plan.md),
decision tree: [human_engine_vs_supervisor](../decisions/human-engine-vs-supervisor.md)
