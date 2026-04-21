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
