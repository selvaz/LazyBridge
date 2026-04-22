# Exporters

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

## Example

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

## OpenTelemetry: what to install, what to wire

`OTelExporter` converts LazyBridge events into OTel spans with the
correct parent/child hierarchy (agent run → tool calls → nested agent
runs).  It requires an OTel SDK install and — in the common case —
an OTLP endpoint to send spans to.

```bash
pip install "lazybridge[otel]"
```

```python
# What this shows: the two ways to hand an OTelExporter its span
# destination. For quick-start: pass an endpoint URL string and
# LazyBridge creates an OTLPSpanExporter internally. For production:
# hand it an exporter instance you've already configured (TLS,
# retries, custom resource attributes) and LazyBridge just wires
# the spans.
# Why both: the endpoint form is the 80% case and keeps OTel
# boilerplate out of app code. The exporter-instance form gives you
# full control without forking the LazyBridge wrapper.

from lazybridge import Agent, Session, OTelExporter

# Form 1 — endpoint string (simplest):
sess_a = Session(exporters=[
    OTelExporter(endpoint="http://otelcol:4318"),   # OTLP HTTP
])

# Form 2 — pre-configured exporter instance (production):
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa
otlp = OTLPSpanExporter(endpoint="http://otelcol:4318",
                        headers={"x-tenant": "acme"})
sess_b = Session(exporters=[
    OTelExporter(exporter=otlp),                    # pre-built
])
```

Span hierarchy emitted by LazyBridge:

```
span: agent.<name>          (AGENT_START → AGENT_FINISH)
├─ span: tool.<tool_name>   (TOOL_CALL → TOOL_RESULT or TOOL_ERROR)
├─ span: model.<model>      (MODEL_REQUEST → MODEL_RESPONSE)
└─ span: agent.<inner_name> (nested agent via as_tool)
```

Usage tokens and cost are attached as span attributes
(`llm.input_tokens`, `llm.output_tokens`, `llm.cost_usd`), so your
tracing backend can aggregate cost per span without a separate
metrics pipeline.  `session.close()` (or the context-manager form)
flushes any in-flight spans — important because OTel batching can
otherwise drop the tail end of a short-lived script.

## Writing a custom exporter

`EventExporter` is a `Protocol` with one required method — shipping
your own is a five-line class.  The reason to write one instead of
using `CallbackExporter` is state: a callback is a pure function;
an exporter class can hold connections, buffers, timers.

```python
# What this shows: a rate-limiting exporter that batches events to
# a remote service. A pure function (CallbackExporter) would have
# nowhere to hold the buffer or the flush timer; a class does.
# Why Protocol-based (not ABC): any object with .export(event:
# dict) -> None works — duck typing keeps third-party exporters
# portable across LazyBridge versions. Add an optional .close()
# method if you buffer; Session.close() will invoke it.

from typing import Any
from lazybridge import Session

class BatchExporter:
    """Buffer events and flush every N or on close()."""

    def __init__(self, sink, batch_size: int = 100) -> None:
        self._sink = sink
        self._buf: list[dict[str, Any]] = []
        self._batch_size = batch_size

    def export(self, event: dict[str, Any]) -> None:
        self._buf.append(event)
        if len(self._buf) >= self._batch_size:
            self._flush()

    def close(self) -> None:
        # Invoked by Session.close() / context-manager exit.
        self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        self._sink(self._buf)
        self._buf.clear()

sess = Session(exporters=[BatchExporter(my_sink, batch_size=50)])
```

Exceptions raised inside `export()` are caught and logged once per
exporter; one broken sink does not stop sibling exporters or the
agent run itself.

## Filtering patterns: `FilteredExporter`

Most dashboards only care about a subset of events.  Wrap any
exporter in `FilteredExporter(inner, event_types={...})` to forward
only those matching the set.  The inner exporter never sees the
rest.

```python
# What this shows: sending all events to a durable JSON log but
# routing only cost-relevant and error events to a dashboard.
# Why compose instead of filter-at-source: LazyBridge emits events
# once; composing exporters decouples the observability stack from
# the agent code. Swap dashboards without touching agents.

from lazybridge import (
    Agent, Session, EventType,
    JsonFileExporter, FilteredExporter, CallbackExporter,
)

def on_dashboard_event(event):
    push_to_dashboard(event)   # hypothetical sink

sess = Session(
    exporters=[
        # Full trail — everything, durable:
        JsonFileExporter("events.jsonl"),
        # Narrow slice — only model cost + errors, pushed to a live dash:
        FilteredExporter(
            CallbackExporter(on_dashboard_event),
            event_types={
                EventType.MODEL_RESPONSE,   # per-call cost
                EventType.TOOL_ERROR,       # failures to alert on
                EventType.AGENT_FINISH,     # end-of-run totals
            },
        ),
    ],
)
```

`event_types=` can be a set of `EventType` enum values or their
string names — both work via `set[str]` comparison on `event["event_type"]`.

## Pitfalls

- Slow exporters block the engine — ``emit`` is synchronous per
  exporter. For high-volume paths, wrap with a queue + worker (or push
  to a log aggregator via ``JsonFileExporter``).
- Exporter exceptions are caught silently; if events don't arrive,
  temporarily wrap with ``CallbackExporter(print)`` to confirm.
- ``StructuredLogExporter`` is a thin wrapper over Python ``logging``
  — it inherits your logger's handlers / format.

!!! note "API reference"

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

!!! warning "Rules & invariants"

    - Each event is a ``dict`` with at minimum ``event_type``, ``session_id``,
      ``run_id`` (possibly ``None``). Agent/engine-specific fields are
      merged in by the emitter.
    - Exporters fire in registration order; an exception in one does NOT
      block others (caught silently — wrap with ``CallbackExporter`` for
      debugging).
    - ``FilteredExporter`` is a combinator — pass an inner exporter and a
      set of event_type strings to forward.
    - ``OTelExporter`` requires ``pip install lazybridge[otel]``.

## See also

[session](session.md),
decision tree: [parallelism](../decisions/parallelism.md)
