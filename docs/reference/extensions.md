# Extension engines & integrations

Framework extensions that live under `lazybridge.ext.*` — `pip install
lazybridge` ships them by default (except `OTelExporter`, which requires the
`[otel]` extra).

> **Connectors moved (0.8).** The MCP connector and the external tool gateway
> are no longer `lazybridge.ext.*` — they moved to the
> [LazyTools](https://tools.lazybridge.com/) package (`lazytools.connectors.{mcp,gateway}`,
> `pip install lazytoolkit`). The old `lazybridge.ext.{mcp,gateway}` import
> paths still work with a `DeprecationWarning` until 0.9.

For narrative usage see the corresponding guides:
[HumanEngine](../guides/mid/human-engine.md),
[SupervisorEngine](../guides/full/supervisor.md),
[MCP](https://tools.lazybridge.com/mcp/),
[Evals](../guides/mid/evals.md),
[OpenTelemetry](../guides/advanced/otel.md),
[Visualizer](../guides/advanced/visualizer.md).

## Human-in-the-loop

::: lazybridge.ext.hil.HumanEngine

::: lazybridge.ext.hil.SupervisorEngine

::: lazybridge.ext.hil.human_agent

::: lazybridge.ext.hil.supervisor_agent

## MCP integration

Moved to `lazytools.connectors.mcp` — see the [MCP guide](https://tools.lazybridge.com/mcp/)
and the [LazyTools overview](https://tools.lazybridge.com/). Install with
`pip install lazytoolkit[mcp]`.

## Evaluation framework

::: lazybridge.ext.evals.EvalSuite

::: lazybridge.ext.evals.EvalCase

::: lazybridge.ext.evals.EvalReport

::: lazybridge.ext.evals.EvalResult

## OpenTelemetry exporter

::: lazybridge.ext.otel.OTelExporter

## Visualizer

::: lazybridge.ext.viz.Visualizer
