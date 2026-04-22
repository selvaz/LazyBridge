# Session & tracing

`Session` is LazyBridge's observability container. It owns an `EventLog`
(SQLite, in-memory by default) and a fan-out of `EventExporter`s that
receive each event — console printers, OpenTelemetry, JSON files,
custom callbacks.

Two ergonomics to remember. `Session(console=True)` installs a
pretty-printing exporter that prints each event to stdout with one line
per event — ideal during development. `Agent("model", verbose=True)`
creates a private `Session(console=True)` for you when you just want to
watch one agent run, no ceremony.

Every Agent constructed with `session=s` auto-registers in
`s.graph` (a `GraphSchema`), and when agents wrap each other as tools
an `as_tool` edge is recorded. You can dump the topology with
`s.graph.to_json()` or `.to_yaml()`.

Call `s.usage_summary()` at the end of a run to get a structured
breakdown: total tokens and cost, per-agent, and per-run.

## Example

```python
from lazybridge import Agent, Session, ConsoleExporter, JsonFileExporter

# Dev — stdout tracing with one flag.
sess = Session(console=True)
Agent("claude-opus-4-7", name="chat", session=sess)("hello")

# Prod — multi-sink observability.
sess = Session(
    db="events.sqlite",
    exporters=[
        JsonFileExporter("events.jsonl"),
        ConsoleExporter(),
    ],
    redact=lambda p: {**p, "task": _mask_pii(p.get("task", ""))},
)
pipeline = Agent.chain(researcher, writer, session=sess)
pipeline("summarise AI trends")

# Observability summary.
summary = sess.usage_summary()
print(summary["total"]["cost_usd"])
print(summary["by_agent"]["researcher"]["input_tokens"])

# Topology for a UI.
print(sess.graph.to_json())
```

## Redaction policy: `redact_on_error`

PII scrubbers fail: a regex throws on unexpected input, a downstream
callable returns `None`, a schema drift surfaces a type the redactor
wasn't written for.  `redact_on_error=` decides what happens when the
redactor itself raises.

```python
# What this shows: two redaction policies for the same pipeline — one
# safe for dev (keep observing even if the redactor breaks), one safe
# for prod (fail closed; never surface unredacted data).
# Why the distinction: silently dropping events hides bugs, but
# silently passing through PII violates privacy policy. Choose
# explicitly.

from lazybridge import Session, JsonFileExporter

def scrub(payload: dict) -> dict:
    # Hypothetical PII scrubber — assume it can raise on bad input.
    return {**payload, "task": mask_pii(payload.get("task", ""))}

# Dev default — warn once, then pass the original payload through so
# you still see events in your logs while debugging the redactor.
dev = Session(
    redact=scrub,
    redact_on_error="fallback",    # the default
    console=True,
)

# Prod — warn once, then DROP the event. No unredacted payload ever
# reaches the EventLog or any exporter.  Fail-closed for compliance.
prod = Session(
    db="events.sqlite",
    exporters=[JsonFileExporter("events.jsonl")],
    redact=scrub,
    redact_on_error="strict",
)
```

Neither mode surfaces the redactor's exception as a run failure — the
agent keeps executing.  The difference is only in what reaches the
observability sink.

## Reading the event log: `session.events.query(...)`

`usage_summary()` aggregates.  `events.query()` is the raw interface —
filter on run_id, event_type, or both, and get back the event records
as dicts.  Use it for ad-hoc analysis, custom dashboards, or
regression tests.

```python
# What this shows: pulling every TOOL_CALL from a specific run to
# verify which tools the engine invoked and what arguments it passed.
# Why query vs reading exporters: exporters are fire-and-forget
# pipes to external sinks. events.query is the canonical in-process
# API for reading your own session's log — thread-safe, indexed on
# run_id + event_type, returns plain dicts.

from lazybridge import Agent, Session, EventType

sess = Session(db="run.sqlite")    # persistent; filter across restarts
agent = Agent("claude-opus-4-7", tools=[...], session=sess)
env = agent("task")

# Narrow by event_type:
tool_calls = sess.events.query(event_type=EventType.TOOL_CALL)

# Narrow by run (Envelope.metadata.run_id is the correlation key):
run_id = env.metadata.run_id
for row in sess.events.query(run_id=run_id, event_type=EventType.TOOL_CALL):
    print(row["payload"]["tool_name"], row["payload"].get("arguments"))

# Full event list for the run, in chronological order:
for row in sess.events.query(run_id=run_id):
    # row keys: "id", "event_type", "run_id", "payload", "ts"
    print(row["ts"], row["event_type"], row["payload"])
```

Event types (from `EventType`):

| Type | Emitted by | Typical payload |
|---|---|---|
| `AGENT_START` / `AGENT_FINISH` | every engine | `agent_name`, `task` (start) or `payload` / `error` (finish) |
| `MODEL_REQUEST` / `MODEL_RESPONSE` | `LLMEngine` | `model`, `input_tokens`, `output_tokens`, `cost_usd` |
| `LOOP_STEP` | `LLMEngine` | turn-level trace in the tool-call loop |
| `TOOL_CALL` / `TOOL_RESULT` / `TOOL_ERROR` | `LLMEngine`, `SupervisorEngine`, `HumanEngine` (errors) | `tool_name`, `arguments`, `result` / `error` |
| `HIL_DECISION` | `HumanEngine`, `SupervisorEngine` | `kind` (continue/retry/store/tool/unknown/empty/input), `command`, truncated `result` |

## Understanding `usage_summary()`

The structure is three aggregation levels on top of the same events —
pick the slice that answers your question.

```python
# What this shows: the shape of the dict sess.usage_summary() returns,
# and which level answers which question.
# Why three levels: a billing question ("what did this pipeline
# cost?") wants "total". An attribution question ("which sub-agent
# burned the tokens?") wants "by_agent". A regression question
# ("did this specific request suddenly get 3x pricier?") wants
# "by_run" keyed on the envelope's run_id.

summary = sess.usage_summary()

# "total": the headline number — one dict with combined counts.
summary["total"] == {
    "input_tokens":  1234,
    "output_tokens": 567,
    "cost_usd":      0.0123,
}

# "by_agent": agent_name → counts. Nested agent-as-tool calls appear
# under their own name, so attribution works across Agent.chain,
# as_tool, Plan — anywhere a child agent was named.
summary["by_agent"] == {
    "researcher": {"input_tokens": 900, "output_tokens": 400, "cost_usd": 0.009},
    "writer":     {"input_tokens": 334, "output_tokens": 167, "cost_usd": 0.0033},
}

# "by_run": run_id → counts + the agent that owned the run. Useful
# when you save run_id alongside request-level application data and
# want to join back to cost later.
summary["by_run"] == {
    "<uuid>": {"agent_name": "researcher", "input_tokens": 900, ...},
    # ...
}
```

Implementation note: `usage_summary()` is `O(events)` with two bulk
queries (AGENT_START + MODEL_RESPONSE), not `O(events * runs)` — it
scales to sessions with thousands of events.

## Custom events via `session.emit()`

Custom engines, custom tools, and app-level code can add their own
events to the log.  Useful for marking business-meaningful
checkpoints that a pure-token-level trace wouldn't capture.

```python
# What this shows: emitting a custom event at an app-level boundary
# (e.g. "user confirmed order") so the event log is the single source
# of truth for both infrastructure and business observability.
# Why: exporters (JsonFile, OTel, custom callbacks) all receive the
# same stream. If your compliance team asks "show every decision this
# pipeline made", emit() keeps that answerable without a second
# logging system.

sess.emit(
    EventType.HIL_DECISION,           # or any EventType; custom strings work too
    {"agent_name": "checkout", "kind": "confirmation", "order_id": 42},
    run_id=env.metadata.run_id,       # optional — correlates with the agent run
)
```

Exporters fire in registration order; if one raises, it's caught and
logged once (so one broken sink doesn't stop the others).  The
`redact` function runs on the payload before either the EventLog or
the exporters see it (see the redaction section above).

## Context-manager lifecycle

`Session.close()` flushes exporters (important for `JsonFileExporter`,
which keeps its file handle open) and closes the EventLog's SQLite
connections.  Skipping it on a long-lived process is fine; on
short-lived scripts use a `with` block to guarantee cleanup.

```python
# Recommended for scripts: guaranteed flush on exit, even if an
# exception propagates. Equivalent to try/finally + sess.close().
with Session(db="run.sqlite", exporters=[JsonFileExporter("events.jsonl")]) as sess:
    Agent.chain(a, b, c, session=sess)("…")
# On exit: exporters flushed + connections closed.
```

## Pitfalls

- ``Session(db=":memory:")`` behaves like ``Session()`` (in-memory).
  Use a filename to persist.
- Exporter failures are caught silently. If an exporter looks like it's
  doing nothing, wrap it in ``CallbackExporter(lambda e: print(e))`` to
  see what's arriving.
- ``Agent(verbose=True)`` creates a **new** Session for that agent; if
  you also pass ``session=another``, ``verbose`` is ignored (the
  explicit session wins).

!!! note "API reference"

    Session(
        *,
        db: str | None = None,            # None = in-memory SQLite
        exporters: list[EventExporter] = None,
        redact: Callable[[dict], dict] | None = None,
        redact_on_error: Literal["fallback", "strict"] = "fallback",
                                          # "fallback" — redactor failure → warn +
                                          #              pass original payload through
                                          # "strict"   — redactor failure → warn +
                                          #              drop the event entirely
        console: bool = False,             # install a ConsoleExporter for stdout tracing
    ) -> Session
    
    session.emit(event_type: EventType, payload: dict, *, run_id: str = None) -> None
    session.add_exporter(exporter: EventExporter) -> None
    session.remove_exporter(exporter: EventExporter) -> None
    session.usage_summary() -> {"total": {...}, "by_agent": {...}, "by_run": {...}}
    
    session.events: EventLog
    session.graph:  GraphSchema          # auto-populated when Agents register
    
    EventLog.record(event_type, payload, *, run_id) -> None
    EventLog.query(*, run_id=None, event_type=None) -> list[dict]
    
    EventType (StrEnum):
      AGENT_START  AGENT_FINISH
      LOOP_STEP
      MODEL_REQUEST  MODEL_RESPONSE
      TOOL_CALL  TOOL_RESULT  TOOL_ERROR
    
    Shortcut: Agent("model", verbose=True) creates a private Session(console=True).

!!! warning "Rules & invariants"

    - Every engine emits events with the same 8-type enum. Hand an Agent
      a ``session=`` and you get a full per-run trace.
    - ``redact`` is called on every payload before recording / exporting;
      use it for PII scrubbing.
    - Nested Agents (Agent A has Agent B as a tool) inherit the outer
      session. All events flow to one EventLog so ``usage_summary()`` can
      aggregate cost across the whole tree.
    - Exporters fire in registration order on every emit. Exceptions raised
      by one exporter do not block others.

## See also

[exporters](exporters.md), [graph_schema](graph-schema.md),
[agent](agent.md)
