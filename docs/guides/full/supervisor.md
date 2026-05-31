# SupervisorEngine

The heavier human-in-the-loop variant. Where
[`HumanEngine`](../mid/human-engine.md) is a single approval prompt,
`SupervisorEngine` is a full REPL: the operator can call tools,
retry registered agents with feedback, inspect store keys, and hand
control back when ready. Drop it into a `Plan` step like any other
engine.

## Signature

```python
from lazybridge import Agent, Tool
from lazybridge.ext.hil import SupervisorEngine, supervisor_agent

# Canonical — Agent + SupervisorEngine
SupervisorEngine(
    *,
    tools=None,                    # list[Tool | Callable | Agent]
    agents=None,                   # list[Agent] — agents the human can `retry`
    store=None,                    # Store the human can `store <key>` to inspect
    input_fn=None,                 # Callable[[str], str] — sync prompt
    ainput_fn=None,                # Callable[[str], Awaitable[str]] — async prompt
    timeout=None,                  # seconds; on expiry triggers default= or raises TimeoutError
    default=None,                  # str returned on timeout
)

agent = Agent(
    engine=SupervisorEngine(
        tools=[search],
        agents=[researcher],
        store=store,
    ),
    name="supervisor",
)


# Sugar — same agent, less plumbing
agent = supervisor_agent(
    tools=[search],
    agents=[researcher],
    store=store,
    name="supervisor",
)
```

The sugar `supervisor_agent(...)` lives in `lazybridge.ext.hil` and
forwards engine kwargs to `SupervisorEngine` and remaining
`**agent_kwargs` to `Agent`. See
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md).

### REPL commands

| Command | Effect |
|---|---|
| `continue [optional text]` | Accept; return optional text as the engine's output. Only terminator. |
| `retry <agent>: <feedback>` | Re-run the named registered agent with feedback appended to the task; the output replaces the supervisor buffer. |
| `store <key>` | Print `store[key]`. |
| `<tool>(<args>)` | Invoke a registered tool with the given arguments. |

Unknown commands print help and re-prompt. Only `continue` ends the
REPL.

## Synopsis

`SupervisorEngine` implements the same `Engine` protocol as
`LLMEngine`, so any agent that accepts an engine can swap one in.
The difference is that the engine's "model" is a human at a REPL
who can inspect, modify, and dispatch — not a one-shot prompt.

The REPL runs on a worker thread so the caller's event loop is not
blocked; `input_fn` is called there. For automated tests, pass a
scripted `input_fn` that returns canned responses. For non-terminal
contexts, pass an `ainput_fn` that wires into your event system.

`agents=` registers Agent instances that the human can `retry`
with feedback. The engine resolves the named agent, appends the
feedback to its task, and runs it; the output replaces the
supervisor's current buffer so subsequent commands operate on the
new value.

## When to use it

- **Interactive debugging of complex pipelines.** Drop in a
  supervisor step after the misbehaving agent; the operator can
  retry it with feedback, inspect store keys, and continue once
  the output is correct.
- **High-stakes operational steps.** A pipeline that posts public
  content, sends emails, or updates production data benefits
  from a supervisor gate where the human can verify the draft,
  retry sub-agents, or call additional tools before approving.
- **Demos and live walkthroughs.** A supervisor mid-pipeline lets
  you steer the demo without restarting the whole run.
- **Sensitive automation under SLA.** An on-call operator can
  intervene when the agent's output looks wrong, without giving
  up the rest of the pipeline's automation.

## When NOT to use it

- **Approval-only flows.** Use
  [`HumanEngine`](../mid/human-engine.md) — it's lighter and
  doesn't pull in REPL machinery the operator doesn't need.
- **Background / unattended pipelines.** A supervisor with
  `timeout=None` (default) hangs forever waiting for input. Pair
  with `timeout=...` + `default="continue"` if you need
  unattended fallback, or use a different engine.
- **Web / HTTP-served agents where the human isn't at a
  terminal.** Pass an `ainput_fn` adapter that wires into your
  request queue / websocket; or build a custom UI on top of
  `HumanEngine`'s simpler protocol.

## Example

```python
from lazybridge import Agent, LLMEngine, Plan, Step, Store, Tool
from lazybridge.ext.hil import SupervisorEngine


def search_web(query: str) -> str:
    """Search the web for ``query``."""
    return "..."


researcher = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    tools=[Tool.wrap(search_web, name="search_web")],
    name="researcher",
)
writer = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="writer",
)


# 1) Standalone supervisor — REPL with tool dispatch + agent retry.
store = Store()
store.write("policy", "publish only peer-reviewed sources")

supervisor = Agent(
    engine=SupervisorEngine(
        tools=[Tool.wrap(search_web, name="search_web")],
        agents=[researcher],          # human can `retry researcher: <feedback>`
        store=store,                  # human can `store policy` to inspect
    ),
    name="supervisor",
)
result = supervisor("draft a policy brief on AI alignment")


# 2) Inside a pipeline — researcher → supervisor → writer.
pipeline = Agent(
    engine=Plan(
        Step(target=researcher, name=researcher.name),
        Step(target=supervisor, name=supervisor.name),
        Step(target=writer,     name=writer.name),
    ),
    name="release-pipeline",
)
pipeline("AI policy brief")


# 3) Scripted inputs for tests.
script = iter([
    "search_web('alignment 2026')",
    "retry researcher: focus on January-April 2026",
    "store policy",
    "continue Final brief approved.",
])

def scripted_input(prompt: str) -> str:
    return next(script)


supervisor_for_test = Agent(
    engine=SupervisorEngine(
        tools=[Tool.wrap(search_web, name="search_web")],
        agents=[researcher],
        store=store,
        input_fn=scripted_input,
    ),
)


# 4) Async UI — wire the prompt into a queue / websocket.
async def web_input(prompt: str) -> str:
    await my_queue.publish({"prompt": prompt})
    return await my_queue.await_response()


supervisor_web = Agent(
    engine=SupervisorEngine(
        tools=[Tool.wrap(search_web, name="search_web")],
        agents=[researcher],
        ainput_fn=web_input,
        timeout=600,
        default="continue",
    ),
    name="supervisor-web",
)
```

## Pitfalls

- **`input_fn` is called from a worker thread.** If it accesses
  thread-unsafe state (like `readline` history), guard it. Async
  callsites should prefer `ainput_fn`.
- **`agents=` expects v1 `Agent` instances.** Duck-typed objects
  work if they expose `__call__` / `run` and a `name` attribute,
  but the supervisor's `retry` command resolves by name — make
  sure the name matches what the human will type.
- **`timeout=None` (default) hangs unattended pipelines forever.**
  Always pair with `timeout=...` + `default=...` for any
  pipeline that might run without a human present.
- **Tool calls in the REPL go via `run_sync`.** Async tool
  functions are driven to completion automatically; this happens
  synchronously inside the REPL so the operator sees the result
  before the next prompt.
- **The REPL terminates only on `continue`.** Other commands
  loop. If your `input_fn` ever returns something the supervisor
  doesn't recognise, it prints help and re-prompts —
  scripted-input tests must end the iterator with `continue`.
- **Session events.** Like any engine, an Agent wrapping
  `SupervisorEngine` emits `AGENT_START` / `AGENT_FINISH` events
  via the session. **Plus**, the REPL emits one `HIL_DECISION`
  event per command — `kind` is one of `continue` / `retry` /
  `store` / `tool` / `unknown` / `empty`, `command` is the raw
  REPL input, and `result` (when present) is a brief of the
  outcome. Auditing a multi-step REPL session is a sequential
  read of those events.
- **`store=` is read-only from the REPL by default.** The
  `store <key>` command prints the value. Writes happen through
  tool calls or registered agents — there's no `store set <key>
  <value>` command (by design — keep mutations explicit and
  traceable).

## See also

- [DeduplicateGuard](../mid/dedup-guard.md) — if worker agents
  accumulate full conversation history across retry loops, attach this
  guard to strip repeated blocks before the LLM sees them.
- [HumanEngine](../mid/human-engine.md) — the lighter
  approval-only variant.
- [Plan](plan.md) — typical container for a supervisor mid-
  pipeline.
- [Store](../mid/store.md) — the inspection target for the
  supervisor's `store <key>` command.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  `supervisor_agent(...)` vs
  `Agent(engine=SupervisorEngine(...))`.
