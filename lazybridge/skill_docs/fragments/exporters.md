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
Three practical stacks:

* **Dev**: `Session(console=True)` (or `Agent(verbose=True)`)
* **Prod**: `exporters=[JsonFileExporter("run.jsonl"), OTelExporter(endpoint=...)]`
* **Custom**: `CallbackExporter(fn)` — pipe events anywhere

Wrap any exporter in `FilteredExporter` to forward only specific event
types, e.g. `{"tool_call", "tool_error"}` for a cost/error dashboard.

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
