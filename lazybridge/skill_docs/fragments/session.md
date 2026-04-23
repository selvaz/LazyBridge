## signature
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

## rules
- Every engine emits events with the same 8-type enum. Hand an Agent
  a ``session=`` and you get a full per-run trace.
- ``redact`` is called on every payload before recording / exporting;
  use it for PII scrubbing.
- Nested Agents (Agent A has Agent B as a tool) inherit the outer
  session. All events flow to one EventLog so ``usage_summary()`` can
  aggregate cost across the whole tree.
- Exporters fire in registration order on every emit. Exceptions raised
  by one exporter do not block others.

## example
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
agents = [researcher, writer]
pipeline = Agent.chain(*agents, session=sess)
pipeline("summarise AI trends")

# Observability summary.
summary = sess.usage_summary()
print(summary["total"]["cost_usd"])
print(summary["by_agent"]["researcher"]["input_tokens"])

# Topology for a UI.
print(sess.graph.to_json())
```

## pitfalls
- ``Session(db=":memory:")`` behaves like ``Session()`` (in-memory).
  Use a filename to persist.
- Exporter failures are caught silently. If an exporter looks like it's
  doing nothing, wrap it in ``CallbackExporter(lambda e: print(e))`` to
  see what's arriving.
- ``Agent(verbose=True)`` creates a **new** Session for that agent; if
  you also pass ``session=another``, ``verbose`` is ignored (the
  explicit session wins).

