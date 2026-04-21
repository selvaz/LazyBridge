## signature
# Protocol
class EventExporter(Protocol):
    def export(self, event: dict) -> None: ...

# Built-ins
CallbackExporter(fn: Callable[[dict], None])
ConsoleExporter(*, stream=sys.stdout)                 # pretty stdout
FilteredExporter(inner: EventExporter, *, event_types: set[str])
JsonFileExporter(path: str)                           # JSONL
StructuredLogExporter(logger_name: str = "lazybridge")
OTelExporter(endpoint: str = None, *, exporter: Any = None)  # OpenTelemetry spans

Usage:
  Session(exporters=[
      ConsoleExporter(),
      JsonFileExporter("events.jsonl"),
      OTelExporter(endpoint="http://jaeger:4318"),
  ])

## rules
- Each event is a ``dict`` with at minimum ``event_type``, ``session_id``,
  ``run_id`` (possibly ``None``). Agent/engine-specific fields are
  merged in by the emitter.
- Exporters fire in registration order; an exception in one does NOT
  block others (caught silently — wrap with ``CallbackExporter`` for
  debugging).
- ``FilteredExporter`` is a combinator — pass an inner exporter and a
  set of event_type strings to forward.
- ``OTelExporter`` requires ``pip install lazybridge[otel]``.

## narrative
Exporters are event sinks: every `Session.emit(...)` call fans out to
them. LazyBridge ships six, covering the common cases (stdout,
logging, JSONL, OpenTelemetry, generic callback, filter combinator),
and `EventExporter` is a trivial Protocol so shipping your own is a
one-method class.

Three practical stacks:

* **Dev**: `Session(console=True)` (or `Agent(verbose=True)`) — one
  `ConsoleExporter` is installed, you read events as they happen.
* **Prod**: `exporters=[JsonFileExporter("run.jsonl"),
  OTelExporter(endpoint=...)]` — durable log + distributed trace.
* **Custom**: `CallbackExporter(push_to_kafka)` — pipe events wherever
  you need.

Filtering: wrap any exporter in a `FilteredExporter` if you only care
about certain event types. Common filter: `{"tool_call", "tool_error"}`
for cost/error dashboards.

## example
```python
from lazybridge import (
    Agent, Session,
    ConsoleExporter, JsonFileExporter, FilteredExporter,
    CallbackExporter, OTelExporter, EventType,
)

def on_error(event):
    if event["event_type"] == EventType.TOOL_ERROR:
        alert_pagerduty(event)

sess = Session(
    db="events.sqlite",
    exporters=[
        JsonFileExporter("run.jsonl"),
        FilteredExporter(
            CallbackExporter(on_error),
            event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
        ),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)

Agent.chain(researcher, writer, session=sess)("…")
```

## pitfalls
- Slow exporters block the engine — ``emit`` is synchronous per
  exporter. For high-volume paths, wrap with a queue + worker (or push
  to a log aggregator via ``JsonFileExporter``).
- Exporter exceptions are caught silently; if events don't arrive,
  temporarily wrap with ``CallbackExporter(print)`` to confirm.
- ``StructuredLogExporter`` is a thin wrapper over Python ``logging``
  — it inherits your logger's handlers / format.

## see-also
[session](session.md),
decision tree: [parallelism](../decisions/parallelism.md)
