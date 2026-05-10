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

`LLMEngine` ships several production-grade knobs that are easy to
miss in the auto-generated signature below. Quick reference:

| Knob | Default | Purpose |
|---|---|---|
| `max_turns` | `20` | Cap on tool-calling rounds; prevents runaway loops |
| `tool_choice` | `"auto"` | `"auto"` / `"any"`; first-turn force-to-call mapped per provider |
| `max_retries` | `3` | Provider transient-error retries with exponential backoff + jitter |
| `request_timeout` | `120.0` | Per-completion deadline. Distinct from `Agent(timeout=N)` (total run). `None` disables |
| `max_parallel_tools` | `8` | Cap on concurrent tool calls within one turn. `None` = unbounded |
| `tool_timeout` | `None` | Per-tool `asyncio.wait_for` deadline; on timeout reports `is_error=True` to the model loop |
| `stream_idle_timeout` | `90.0` | Idle gap between streaming chunks before `StreamStallError`; pass `None` to disable (a one-shot `UserWarning` is emitted at `LLMEngine.__init__` time, not at stream time). |
| `stream_buffer` | `64` | Bounded queue for streaming producers. Must be ≥1 |
| `allow_dangerous_native_tools` | `False` | Security gate for `CODE_EXECUTION` / `COMPUTER_USE`; opt-in required |
| `thinking` | `False` / `ThinkingConfig` | Extended-thinking opt-in.  Anthropic Opus 4.6+ / Claude 4.7 use adaptive thinking (server-managed budget; pass `display="omitted"` to hide thoughts).  OpenAI `o1`/`o3`/`o4`/`gpt-5` and Gemini 2.5+ surface `reasoning_tokens` automatically; passing a `ThinkingConfig(effort=...)` is forwarded where the provider supports it.  See provider-capability matrix in `lazybridge.matrix`. |
| `strict_multimodal` | `False` | Raise `UnsupportedFeatureError` when the model doesn't support an attachment modality |

`strict_native_tools` is **not** an `LLMEngine` knob — it lives on
`BaseProvider` (constructor arg + class attribute). Configure it
where you build the provider, e.g. `AnthropicProvider(...,
strict_native_tools=True)`; LLMEngine reads it off the resolved
provider at request time. See
[`BaseProvider`](../guides/advanced/base-provider.md).

::: lazybridge.LLMEngine

## Plan + Step

::: lazybridge.Plan

::: lazybridge.Step

## Engine errors

::: lazybridge.PlanCompileError

::: lazybridge.PlanRuntimeError

::: lazybridge.PlanPaused

::: lazybridge.ConcurrentPlanRunError

::: lazybridge.ToolTimeoutError

::: lazybridge.StreamStallError
