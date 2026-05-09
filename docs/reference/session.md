# Session & observability

`Session` is the event bus that fans observability events into
exporters and exposes the `GraphSchema` topology view. Six core
exporter classes ship under `lazybridge.*`; `OTelExporter` lives
under `lazybridge.ext.otel` (see [Extension engines](extensions.md)).

For narrative usage see [Guides → Mid → Session](../guides/mid/session.md),
[Guides → Full → Exporters](../guides/full/exporters.md), and
[Guides → Full → GraphSchema](../guides/full/graph-schema.md).

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
