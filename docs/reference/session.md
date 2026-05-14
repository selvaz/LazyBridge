# Session & observability

`Session` is the event bus that fans observability events into
exporters and exposes the `GraphSchema` topology view. Six core
exporter classes ship under `lazybridge.*`; `OTelExporter` lives
under `lazybridge.ext.otel` (see [Extension engines](extensions.md)).

For narrative usage see [Guides → Mid → Session](../guides/mid/session.md),
[Guides → Full → Exporters](../guides/full/exporters.md), and
[Guides → Full → GraphSchema](../guides/full/graph-schema.md).

| Symbol | Role |
|---|---|
| `Session` | The event bus.  Attached to an Agent via `session=` (or implicitly via `verbose=True`). |
| `EventLog` | SQLite-backed event store under `Session.events`. |
| `EventType` | StrEnum of emitted event kinds (`AGENT_START`, `TOOL_CALL`, `AGENT_FINISH`, …). |
| `GraphSchema` | Topology view of registered agents + tool edges; renderable via `Session.graph.to_json()`. |
| `EventExporter` | Base protocol — implement `export(event)` to add your own sink. |
| `ConsoleExporter` | Human-readable stdout exporter; used implicitly by `verbose=True`. |
| `CallbackExporter` | Routes events to a user-supplied callable. |
| `FilteredExporter` | Wrap any exporter to drop events that don't match a predicate. |
| `JsonFileExporter` | Newline-delimited JSON sink. |
| `StructuredLogExporter` | Logs events as structured records via the `logging` module. |
| `OTelExporter` (ext) | OpenTelemetry GenAI conventions; see [Guides → Advanced → OpenTelemetry](../guides/advanced/otel.md). |

## Session

::: lazybridge.Session

## Event log + types

::: lazybridge.EventLog

::: lazybridge.EventType

## Graph topology

::: lazybridge.GraphSchema

## Exporters

::: lazybridge.EventExporter

::: lazybridge.CallbackExporter

::: lazybridge.ConsoleExporter

::: lazybridge.FilteredExporter

::: lazybridge.JsonFileExporter

::: lazybridge.StructuredLogExporter
