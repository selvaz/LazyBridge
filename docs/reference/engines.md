# Engines

`LLMEngine` is the LLM-driven tool-calling loop; `Plan` is the
deterministic-DAG engine; `Step` is its unit. `PlanCompileError`
fires at construction for invalid DAGs; `ToolTimeoutError` and
`StreamStallError` surface from the LLM engine's safety nets.

For narrative usage see [Guides → Full → Plan](../guides/full/plan.md),
[Step](../guides/full/step.md), and the
[Engine protocol](../guides/advanced/engine-protocol.md) (extension
surface).

## LLM engine

::: lazybridge.LLMEngine

## Plan + Step

::: lazybridge.Plan

::: lazybridge.Step

## Engine errors

::: lazybridge.PlanCompileError

::: lazybridge.ToolTimeoutError

::: lazybridge.StreamStallError
