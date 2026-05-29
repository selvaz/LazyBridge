# Providers

The catalogue of LLM providers shipped with LazyBridge, the tier
aliases each one resolves, and the per-provider quirks (thinking
modes, native tools, deprecation timelines). For writing a brand-new
provider see [BaseProvider](base-provider.md).

> **Pricing and model lineup snapshot from late 2025.** LLM provider
> economics shift fast — treat the tables below as a structural
> reference (which alias resolves to which model, which features
> work on which model) rather than as live pricing.

## Signature

```python
from lazybridge import Agent, LLMEngine

# Direct model selection — provider inferred from the model string.
Agent(engine=LLMEngine("claude-opus-4-8"))
Agent(engine=LLMEngine("gpt-5.4-mini"))

# Tier-based selection — model never appears in app code.
Agent.from_provider("anthropic", tier="top")     # → claude-opus-4-8
Agent.from_provider("openai",    tier="medium")  # → gpt-5.4-mini
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
| `top` | `claude-opus-4-8` | 1 M | 128 K | $5.00 | $25.00 |
| `expensive` | `claude-opus-4-7` | 1 M | 128 K | $5.00 | $25.00 |
| `medium` | `claude-sonnet-4-6` | 1 M | 64 K | $3.00 | $15.00 |
| `cheap` | `claude-haiku-4-5` | 200 K | 64 K | $1.00 | $5.00 |
| `super_cheap` | `claude-3-haiku` | 200 K | 4 K | $0.25 | $1.25 |

- **Thinking.** `opus-4-8` / `opus-4-7` / `opus-4-6` / `sonnet-4-6` use adaptive
  thinking (no `budget_tokens` argument). `haiku-4-5` and earlier
  3.x models require `ThinkingConfig(budget_tokens=N)`. `opus-4-8` and `opus-4-7`
  do **not** accept `temperature`.
- **Native tools.** `WEB_SEARCH`, `CODE_EXECUTION`, `COMPUTER_USE`.

### OpenAI

| tier | model | ctx | max_out | $/M in | $/M cached | $/M out |
|---|---|---|---|---|---|---|
| `top` | `gpt-5.5-pro` | 1 M | 128 K | $30.00 | — | $180.00 |
| `expensive` | `gpt-5.5` | 1 M | 128 K | $5.00 | $0.50 | $30.00 |
| `medium` | `gpt-5.4-mini` | 400 K | 128 K | $0.75 | $0.075 | $4.50 |
| `cheap` | `gpt-5.4-nano` | 400 K | 128 K | $0.20 | $0.02 | $1.25 |
| `super_cheap` | `gpt-4o-mini` | 128 K | 16 K | $0.15 | — | $0.60 |

Other supported models (passed verbatim, no tier alias):
`gpt-5.4-pro` ($30 / $180), `gpt-5.4` ($2.50 / $0.25 cache / $15),
`gpt-5` ($1.25 / $10), `gpt-4o` ($2.50 / $10), `gpt-4.1` ($2 / $8),
`gpt-4.1-mini` ($0.40 / $1.60), `o3` ($2 / $8), `o4-mini`
($1.10 / $4.40).

- **Thinking.** `gpt-5.5` / `gpt-5.5-pro` accept
  `reasoning_effort ∈ {none, low, medium, high, xhigh}` (default
  `medium`). The `o`-series and `gpt-5.4-pro` accept
  `reasoning_effort ∈ {low, medium, high}`. Standard GPT models
  don't support thinking.
- **Native tools.** `WEB_SEARCH`, `CODE_EXECUTION`, `FILE_SEARCH`,
  `COMPUTER_USE`, `IMAGE_GENERATION`.
- **Cache.** Automatic via `prompt_tokens_details.cached_tokens`;
  `cached_input` rate applied when published (`gpt-5.5`,
  `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`).
- **Long-context surcharge** (>272K input on `gpt-5.x`) is **not**
  modeled in cost rollup — the reported cost may under-count for
  large prompts.

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

| tier | model | ctx | max_out | $/M in | $/M cached | $/M out |
|---|---|---|---|---|---|---|
| `top` / `expensive` | `deepseek-v4-pro` | 1 M | 384 K | $1.74 | $0.145 | $3.48 |
| `medium` / `cheap` / `super_cheap` | `deepseek-v4-flash` | 1 M | 384 K | $0.14 | $0.028 | $0.28 |

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
- **`gpt-5.5-mini` / `gpt-5.5-nano` don't exist** yet; the
  `medium` and `cheap` tiers stay on `gpt-5.4-mini` /
  `gpt-5.4-nano` until OpenAI ships them.
- **`gpt-5-mini` doesn't exist either.** The current OpenAI
  `mini` variant is `gpt-5.4-mini`.
- **`gemini-2.0-flash` deprecation** lands June 1 2026; switch to
  `gemini-2.5-flash-lite` before then.
- **Adaptive thinking ignores `budget_tokens`.** Anthropic
  `claude-opus` / `claude-sonnet` 4.6+ pick their own thinking
  budget; passing `ThinkingConfig(budget_tokens=...)` is
  no-effect.
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
