# Extension engines & integrations

Surface that lives under `lazybridge.ext.*`. These are first-class
extensions — `pip install lazybridge` ships them by default
(except `OTelExporter` which requires the `[otel]` extra).

For narrative usage see the corresponding guides:
[HumanEngine](../guides/mid/human-engine.md),
[SupervisorEngine](../guides/full/supervisor.md),
[MCP](../guides/mid/mcp.md),
[Evals](../guides/mid/evals.md),
[OpenTelemetry](../guides/advanced/otel.md),
[Visualizer](../guides/advanced/visualizer.md).

## Human-in-the-loop

::: lazybridge.ext.hil.HumanEngine

::: lazybridge.ext.hil.SupervisorEngine

::: lazybridge.ext.hil.human_agent

::: lazybridge.ext.hil.supervisor_agent

## MCP integration

::: lazytools.connectors.mcp.MCP

::: lazytools.connectors.mcp.MCPServer

## Evaluation framework

::: lazybridge.ext.evals.EvalSuite

::: lazybridge.ext.evals.EvalCase

::: lazybridge.ext.evals.EvalReport

::: lazybridge.ext.evals.EvalResult

## OpenTelemetry exporter

::: lazybridge.ext.otel.OTelExporter

## Visualizer

::: lazybridge.ext.viz.Visualizer
