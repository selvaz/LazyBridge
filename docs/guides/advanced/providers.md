# Providers

The catalogue of LLM providers shipped with LazyBridge, the tier
aliases each one resolves, and the per-provider quirks (thinking
modes, native tools, deprecation timelines). For writing a brand-new
provider see [BaseProvider](base-provider.md).

> **Pricing and model lineup snapshot from July 2026.** LLM provider
> economics shift fast — treat the tables below as a structural
> reference (which alias resolves to which model, which features
> work on which model) rather than as live pricing.

## Signature

```python
from lazybridge import Agent, LLMEngine

# Direct model selection — provider inferred from the model string.
Agent(engine=LLMEngine("claude-opus-5"))
Agent(engine=LLMEngine("gpt-5.6-luna"))

# Tier-based selection — model never appears in app code.
Agent.from_provider("anthropic", tier="top")     # → claude-fable-5
Agent.from_provider("openai",    tier="medium")  # → gpt-5.6-luna
Agent.from_provider("google",    tier="cheap")   # → gemini-3.1-flash-lite-preview
```

`Agent.from_provider` is sugar for
`Agent(engine=LLMEngine(<resolved-model>, provider=<name>))`. See
[Canonical vs sugar](../../concepts/canonical-vs-sugar.md) for the
breakdown.

### Tier names

| Tier | Intent |
|---|---|
| `super_cheap` | Smallest / cheapest model in the lineup; for parsing, classification, throwaway calls |
| `cheap` | Default budget tier |
| `medium` | The default for `Agent.from_provider(...)` |
| `expensive` | Premium reasoning / long-context tier |
| `top` | The flagship model |

Each provider's `_TIER_ALIASES` table maps these strings to a concrete
model name. A string not in the table is treated as a literal model
name (passthrough).

## Built-in providers

### Anthropic

| tier | model | ctx | max_out | $/M in | $/M out |
|---|---|---|---|---|---|
| `top` | `claude-fable-5` | 1 M | 128 K | $10.00 | $50.00 |
| `expensive` | `claude-opus-5` | 1 M | 128 K | $5.00 | $25.00 |
| `medium` | `claude-sonnet-5` | 1 M | 128 K | $2.00¹ | $10.00¹ |
| `cheap` | `claude-haiku-4-5` | 200 K | 64 K | $1.00 | $5.00 |
| `super_cheap` | `claude-3-haiku` | 200 K | 4 K | $0.25 | $1.25 |

¹ Sonnet 5 introductory pricing through 2026-08-31; rises to
$3.00 / $15.00 per million tokens after that date.

Not tier-aliased: `claude-mythos-5` ($10.00 / $50.00, same underlying
model as Fable 5 with fewer safety guardrails) — restricted to vetted
partners (Project Glasswing / US government cyber defenders), not
reachable with an ordinary API key. Older pinned ids
(`claude-opus-4-8`, `claude-opus-4-7`, `claude-opus-4-6`,
`claude-sonnet-4-6`, `claude-opus-4-1`, …) still resolve — see
`_PRICE_TABLE` in `core/providers/anthropic.py` for the full list.

- **Thinking.** `fable-5` / `mythos-5` / `opus-5` / `opus-4-8` /
  `opus-4-7` / `opus-4-6` / `sonnet-5` / `sonnet-4-6` use adaptive
  thinking (no `budget_tokens` argument). `haiku-4-5` and earlier
  3.x models require `ThinkingConfig(budget_tokens=N)`.
  `fable-5` / `mythos-5` / `opus-5` / `opus-4-8` / `opus-4-7` /
  `sonnet-5` do **not** accept `temperature` / `top_p` / `top_k`.
- **Effort.** `fable-5`, `mythos-5`, `opus-5`, `opus-4-8`, `opus-4-7`,
  `opus-4-6`, `sonnet-5`, `sonnet-4-6`, and `opus-4-5` accept
  `ThinkingConfig(effort=...)` ∈ `{low, medium, high, xhigh, max}`
  (default `high`), sent as `output_config.effort`. It works with or
  without `thinking` enabled and, unlike `budget_tokens`, shapes *all*
  response tokens (text, tool calls, thinking). `xhigh` isn't available
  on `opus-4-6` / `sonnet-4-6` / `opus-4-5` — LazyBridge downgrades an
  `xhigh` request on those models to `max` with a warning. Shorthand:
  `LLMEngine(model, thinking="low")`.
- **Native tools.** `WEB_SEARCH`, `CODE_EXECUTION`, `COMPUTER_USE`.

### OpenAI

| tier | model | ctx | max_out | $/M in | $/M cached | $/M out |
|---|---|---|---|---|---|---|
| `top` | `gpt-5.6-sol` | 1.05 M | 128 K | $5.00 | $0.50 | $30.00 |
| `expensive` | `gpt-5.6-terra` | 1.05 M | 128 K | $2.50 | $0.25 | $15.00 |
| `medium` | `gpt-5.6-luna` | 1.05 M | 128 K | $1.00 | $0.10 | $6.00 |
| `cheap` | `gpt-5.4-nano` | 400 K | 128 K | $0.20 | $0.02 | $1.25 |
| `super_cheap` | `gpt-4o-mini` | 128 K | 16 K | $0.15 | — | $0.60 |

GPT-5.6 (released 2026-07-09) replaced the old flagship+`-pro` shape
with three tiers: Sol (best coding / hardest reasoning, OpenAI's
"workhorse"), Terra (balanced general flagship), Luna (fast/light).
The bare alias `gpt-5.6` routes to Sol. GPT-5.6 also introduces
explicit prompt-cache breakpoints and a 30-minute minimum cache life.

Other supported models (passed verbatim, no tier alias):
`gpt-5.5-pro` ($30 / $180), `gpt-5.5` ($5 / $0.50 cache / $30),
`gpt-5.4-pro` ($30 / $180), `gpt-5.4` ($2.50 / $0.25 cache / $15),
`gpt-5.4-mini` ($0.75 / $0.075 cache / $4.50), `gpt-5`
($1.25 / $10), `gpt-4o` ($2.50 / $10), `gpt-4.1` ($2 / $8),
`gpt-4.1-mini` ($0.40 / $1.60), `o3` ($2 / $8), `o4-mini`
($1.10 / $4.40).

- **Thinking.** `gpt-5.6-*` / `gpt-5.5` / `gpt-5.5-pro` accept
  `reasoning_effort ∈ {none, low, medium, high, xhigh}` (default
  `medium`). The `o`-series and `gpt-5.4-pro` accept
  `reasoning_effort ∈ {low, medium, high}`. Standard GPT models
  don't support thinking. Shorthand: `LLMEngine(model, thinking="low")`
  — mirrors the Anthropic `ThinkingConfig.effort` shorthand, since both
  providers read the same `ThinkingConfig.effort` field.
- **Native tools.** `WEB_SEARCH`, `CODE_EXECUTION`, `FILE_SEARCH`,
  `COMPUTER_USE`, `IMAGE_GENERATION`.
- **Cache.** Automatic via `prompt_tokens_details.cached_tokens`;
  `cached_input` rate applied when published (`gpt-5.6-*`, `gpt-5.5`,
  `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`).
- **Long-context surcharge** (>272K input on `gpt-5.x`) is **not**
  modeled in cost rollup — the reported cost may under-count for
  large prompts.
- **Not modeled.** OpenAI's realtime voice models (`GPT-Live-1`,
  `GPT-Live-1 mini`, released 2026-07-08) are a separate full-duplex
  Realtime API, not the text Responses/Chat Completions path
  `OpenAIProvider` wraps — out of scope for this provider.

### Google

| tier | model | ctx | max_out | $/M in | $/M out |
|---|---|---|---|---|---|
| `top` | `gemini-3.1-pro-preview` | 1 M | 64 K | $2.00 | $12.00 |
| `expensive` | `gemini-2.5-pro` | 1 M | 64 K | $1.25 | $10.00 |
| `medium` | `gemini-3-flash-preview` | 1 M | 64 K | $0.50 | $3.00 |
| `cheap` | `gemini-3.1-flash-lite-preview` | 1 M | 64 K | $0.25 | $1.50 |
| `super_cheap` | `gemini-2.5-flash-lite` | 1 M | 64 K | $0.10 | $0.40 |

- **Thinking.** `gemini-3.x` accepts
  `ThinkingConfig(thinking_level=...)` with `low` / `medium` /
  `high`. `gemini-2.x` accepts `ThinkingConfig(thinking_budget=N)`;
  `-1` selects auto-budget.
- **Native tools.** `GOOGLE_SEARCH`, `WEB_SEARCH`, `GOOGLE_MAPS`.
- **Warning.** Google Search + structured output produces a
  provider 400 — they're mutually exclusive.
- **Deprecation.** `gemini-2.0-flash` retires June 1 2026; do not
  use in new code.

### DeepSeek

Prices below are peak-hour rates (01:00-04:00 and 06:00-10:00 UTC); LazyBridge
always costs at the peak rate for a conservative estimate. Off-peak rates are
exactly half.

| tier | model | ctx | max_out | $/M in | $/M cached | $/M out |
|---|---|---|---|---|---|---|
| `top` / `expensive` | `deepseek-v4-pro` | 1 M | 384 K | $1.32 | $0.044 | $3.96 |
| `medium` / `cheap` / `super_cheap` | `deepseek-v4-flash` | 1 M | 384 K | $0.44 | $0.014 | $1.32 |

- **Thinking.** Both V4 models accept `ThinkingConfig` →
  `reasoning_content` field on the response. In thinking mode the
  provider strips `temperature` / `top_p` / `presence_penalty` /
  `frequency_penalty`. `ThinkingConfig` on non-V4 models raises
  `ValueError`.
- **Cache.** Automatic on repeated prefixes ≥1024 tokens; no
  opt-in required.
- **Native tools.** None (function calling is supported).
- **Deprecation (retire 2026-07-24).** `deepseek-reasoner` and
  `deepseek-chat` both alias to `deepseek-v4-flash`.

### LMStudio

A local OpenAI-compatible runtime. `LMStudioProvider` extends
`OpenAIProvider`; point `OPENAI_BASE_URL` at your LM Studio
instance and use any model name your local install serves.

### LiteLLM

The unified bridge for the long tail (Mistral, Cohere, Groq,
Bedrock, Vertex, Ollama, etc.). Use the `litellm/` model-string
prefix to route through `LiteLLMProvider`. Native providers
(Anthropic, OpenAI, Google, DeepSeek) still handle their own
models directly — LiteLLM is the catch-all for the rest.

```python
Agent(engine=LLMEngine("litellm/groq/llama-3.3-70b"))
```

## `tool_choice` values

LLMEngine accepts a `tool_choice=` kwarg that drives provider tool
selection:

| Value | Meaning |
|---|---|
| `"auto"` | Model decides (default) |
| `"none"` | No tool calls allowed |
| `"required"` | Must call at least one tool |
| `"any"` | Alias for `"required"`; mapped to provider equivalent (`"required"` for OpenAI, `{"type":"required"}` for Anthropic) |
| `"<tool_name>"` | Must call the named tool |

After the first tool-call turn, `tool_choice` resets to `"auto"`
automatically — so a forced first invocation doesn't lock the rest
of the loop.

DeepSeek does **not** support `tool_choice` in thinking mode.

## Google `finish_reason` mapping

The Google provider normalises `finish_reason` strings so callers
don't have to switch on Gemini-specific values:

| Gemini value | Normalised |
|---|---|
| `MAX_TOKENS` | `"max_tokens"` |
| `SAFETY` / `RECITATION` / `BLOCKLIST` / `PROHIBITED_CONTENT` / `SPII` | `"stop"` |
| anything else | `"end_turn"` |

## Pitfalls

- **DeepSeek tier collapse.** Three of the five tier aliases
  (`medium` / `cheap` / `super_cheap`) all map to
  `deepseek-v4-flash` — there's no smaller model in the lineup.
- **`gpt-5.6-nano` doesn't exist.** The `cheap` tier stays on
  `gpt-5.4-nano` — Luna is the fast/light GPT-5.6 tier but isn't
  actually cheaper per-token than 5.4-nano.
- **`gemini-2.0-flash` deprecation** lands June 1 2026; switch to
  `gemini-2.5-flash-lite` before then.
- **Adaptive thinking ignores `budget_tokens`.** Anthropic
  `claude-fable`/`mythos`/`opus` 5, `claude-opus` / `claude-sonnet`
  4.6+ pick their own thinking budget; passing
  `ThinkingConfig(budget_tokens=...)` is no-effect. Use
  `ThinkingConfig(effort=...)` instead.
- **`claude-mythos-5` is access-restricted**, not a public tier —
  don't wire it into `Agent.from_provider("anthropic", tier=...)`.
- **`tool_choice="any"` is not passed literally.** It maps to
  `"required"` (or the provider equivalent) at request time.
- **Pricing changes faster than these tables.** Check the
  provider's current rate card before reasoning about cost in
  production.

## See also

- [BaseProvider](base-provider.md) — write your own provider
  when none of the built-ins fits.
- [Native tools](../basic/native-tools.md) — what each provider
  exposes server-side; the per-provider table above lists the
  supported `NativeTool` enum values.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  `Agent.from_provider("…", tier="top")` is one of the few
  factory methods that's not pure sugar (it builds the engine
  with the tier alias and an explicit `provider=`).
