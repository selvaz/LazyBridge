# Exporters

Three practical stacks:

* **Dev**: `Session(console=True)` (or `Agent(verbose=True)`).
* **Prod**: `JsonFileExporter(path="run.jsonl")` for durable logs +
  `OTelExporter(endpoint=...)` for distributed tracing, both behind
  `Session(batched=True)`.
* **Custom**: `CallbackExporter(fn=fn)` — pipe events anywhere
  (Slack alerts, Prometheus, your own DB).

Wrap any exporter in `FilteredExporter` to forward only specific
event types — e.g. `{"tool_error", "agent_finish"}` for a cost+error
dashboard.

## Example

```python
from lazybridge import (
    Agent, Session,
    ConsoleExporter, JsonFileExporter, FilteredExporter,
    CallbackExporter, EventType,
)
from lazybridge.ext.otel import OTelExporter

def on_alert(event):
    if event["event_type"] == EventType.TOOL_ERROR:
        alert_pagerduty(event)

sess = Session(
    db="events.sqlite",
    batched=True,                          # non-blocking emit
    exporters=[
        JsonFileExporter(path="run.jsonl"),
        FilteredExporter(
            CallbackExporter(fn=on_alert),
            event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
        ),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)

Agent.chain(researcher, writer, session=sess)("…")
sess.flush()                               # drain the writer before exit
```

## Pitfalls

- Slow exporters block the engine when ``Session(batched=False)``
  (the default). Set ``batched=True`` for any exporter doing network
  I/O.
- Exporter exceptions warn once per instance and are suppressed
  afterwards. If only the first failure shows up, wrap with
  ``CallbackExporter(fn=print)`` while debugging.
- ``OTelExporter`` keeps a per-instance tracer rooted in its own
  ``TracerProvider`` so multiple exporters in one process don't
  fight. The provider is also installed globally as a best-effort
  default; you can supply your own and pass an in-memory exporter
  for tests.
- When ``Session.batched=True``, ``session.events.query(...)`` may
  return stale rows until ``session.flush()`` drains the writer.

!!! note "API reference"

    # Protocol
    class EventExporter(Protocol):
        def export(self, event: dict) -> None: ...
        # Optional: close() is called by Session.close() when present.
    
    # Built-ins shipped from ``lazybridge`` (core).
    CallbackExporter(fn: Callable[[dict], None])
    ConsoleExporter(*, stream=sys.stdout)            # pretty stdout
    FilteredExporter(inner: EventExporter, *, event_types: set[str])
    JsonFileExporter(path: str)                       # JSONL append
    StructuredLogExporter(logger_name: str = "lazybridge")
    
    # Built-in shipped from ``lazybridge.ext.otel`` (alpha extension).
    from lazybridge.ext.otel import OTelExporter
    OTelExporter(endpoint: str | None = None, *, exporter: Any | None = None)
    
    Usage:
      Session(exporters=[
          ConsoleExporter(),
          JsonFileExporter(path="events.jsonl"),
          OTelExporter(endpoint="http://otelcol:4318"),
      ])

!!! warning "Rules & invariants"

    - Each event is a ``dict`` with at minimum ``event_type``,
      ``session_id``, ``run_id`` (possibly ``None``). Engine-specific
      fields are merged in by the emitter.
    - Exporters fire in registration order. An exception in one exporter
      does NOT block others; LazyBridge warns once per exporter instance
      and suppresses subsequent failures from the same exporter.
    - ``FilteredExporter`` is a combinator — pass an inner exporter and a
      set of event_type strings to forward.
    - ``OTelExporter`` requires ``pip install lazybridge[otel]``.  It emits
      spans conforming to the OpenTelemetry GenAI Semantic Conventions
      (``gen_ai.system``, ``gen_ai.usage.input_tokens``,
      ``gen_ai.tool.call.id``, …) so dashboards built for the standard
      render LazyBridge traces without translation.
    - OTel span hierarchy mirrors the run:
        * ``invoke_agent <name>`` (root per ``Agent.run``)
        * ├ ``chat <model>`` (one per LLM round-trip)
        * └ ``execute_tool <tool>`` (one per tool invocation, correlated
          by ``tool_use_id``)
      Cross-agent parenting works automatically through OTel contextvars
      — an inner Agent invoked through a tool becomes a descendant of the
      outer tool span, no run-id chaining required.
    - For high-throughput emit paths, pair ``Session(batched=True,
      on_full="hybrid")`` with the slower exporters (OTel, JSON-file).

## See also

- [Session](session.md) — the bus that fans events into exporters.
- [GraphSchema](graph-schema.md) — topology view, separate from events.
