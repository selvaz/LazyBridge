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

`LLMEngine` ships eight production-grade knobs that are easy to miss
in the auto-generated signature below. Quick reference:

| Knob | Default | Purpose |
|---|---|---|
| `max_turns` | `20` | Cap on tool-calling rounds; prevents runaway loops |
| `tool_choice` | `"auto"` | `"auto"` / `"any"`; first-turn force-to-call mapped per provider |
| `max_retries` | `3` | Provider transient-error retries with exponential backoff + jitter |
| `request_timeout` | `120.0` | Per-completion deadline. Distinct from `Agent(timeout=N)` (total run). `None` disables |
| `max_parallel_tools` | `8` | Cap on concurrent tool calls within one turn. `None` = unbounded |
| `tool_timeout` | `None` | Per-tool `asyncio.wait_for` deadline; on timeout reports `is_error=True` to the model loop |
| `stream_idle_timeout` | `90.0` | Idle gap between streaming chunks before `StreamStallError`; `None` disables (one-shot warning) |
| `stream_buffer` | `64` | Bounded queue for streaming producers. Must be ≥1 |
| `allow_dangerous_native_tools` | `False` | Security gate for `CODE_EXECUTION` / `COMPUTER_USE`; opt-in required |
| `strict_native_tools` | `False` | Raise `UnsupportedNativeToolError` when an unsupported native tool is requested (vs the warning-and-drop default) |
| `strict_multimodal` | `False` | Raise `UnsupportedFeatureError` when the model doesn't support an attachment modality |

::: lazybridge.LLMEngine

## Plan + Step

::: lazybridge.Plan

::: lazybridge.Step

## Engine errors

::: lazybridge.PlanCompileError

::: lazybridge.ToolTimeoutError

::: lazybridge.StreamStallError
