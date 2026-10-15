# Providers and models

Read when choosing a model string, a tier alias, thinking/effort, prompt
caching, or writing a custom provider. Source of truth:
`lazybridge/core/providers/*.py` (`_TIER_ALIASES`, `_PRICE_TABLE`,
`_FALLBACKS`), `lazybridge/engines/llm.py` (provider registry),
`lazybridge/core/types.py` (`ThinkingConfig`, `CacheConfig`); docs
`docs/guides/advanced/{providers,base-provider}.md`, `docs/reference/providers.md`.

## Model strings route to providers

`LLMEngine("claude-opus-5-5")` infers the provider from the string;
`LLMEngine("top", provider="anthropic")` resolves a tier. Built-in providers:
Anthropic, OpenAI, Google, DeepSeek, LM Studio (local), LiteLLM (any
LiteLLM-routed model). Unknown strings raise `ValueError` (no silent default
provider since 0.7.9). API keys come from the usual provider env vars.

## Tier aliases (LazyBridge 1.8.2)

`Agent.from_provider(provider, tier="medium")` or
`LLMEngine(tier, provider=...)`; tiers: `super_cheap`, `cheap`, `medium`,
`expensive`, `top` (type `Tier`). A string not in the table passes through as
a literal model id.

| Provider | top | expensive | medium | cheap | super_cheap |
|---|---|---|---|---|---|
| anthropic | claude-fable-5-1 | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-4-5 | claude-haiku-4-5 |
| openai | gpt-6-astra | gpt-6.1-sol | gpt-6.1-sol | gpt-6-luna | gpt-6-luna |
| google | gemini-3.1-pro-preview | gemini-2.5-pro | gemini-3-flash-preview | gemini-3.1-flash-lite-preview | gemini-2.5-flash-lite |
| deepseek | deepseek-v4-pro | deepseek-v4-pro | deepseek-flash | deepseek-flash | deepseek-flash |
| lmstudio | gpt-oss-20b | gemma-4-26b-a4b | qwen3.5-14b | qwen3.5-4b | gemma-4-e2b |

The table rots: re-check `Provider._TIER_ALIASES` in the installed version
before quoting it. Use a tier only when "current best in tier" is what the
caller wants; pin and validate an explicit id when behaviour must not drift.
`_FALLBACKS` entries are metadata; automatic model fallback is **not**
implemented (use `Agent(fallback=...)`). Restricted models
(`claude-mythos-*`, `gpt-5.6-cyber`) are priced but never tier-aliased.

These catalogue tiers are unrelated to the coding engines' model aliases
(`sonnet`/`opus`/`haiku` for Claude Code, runtime config for Codex) and to
LazyTools' quota-routing tiers (`basic`/`writing`/`thinking`/`critical`).

## Thinking and effort

- `LLMEngine(model, thinking=True)`, `thinking="low"` (effort shorthand) or
  `thinking=ThinkingConfig(effort="high", budget_tokens=None, display=None)`
  from `lazybridge.core.types`. Effort values `low/medium/high/xhigh/max`;
  OpenAI maps `max` to its own top value; Anthropic sends
  `output_config.effort` and downgrades unsupported `xhigh` with a warning.
- Opus 5.5 / Fable 5.1 / Mythos 5.1 reject forced `tool_choice`; LazyBridge
  sends `"auto"` with a `UserWarning`.
- Claude 5-family models reject `temperature`/`top_p`/`top_k`.

## Caching and cost

- `LLMEngine(..., cache=True)` (or `Agent(cache=True)`) marks the static
  prefix cacheable. The dataclass `CacheConfig` must be imported from
  `lazybridge.core.types` (top-level re-export is deprecated).
- `Envelope.metadata.cost_usd` comes from the price tables; long-context
  surcharges and peak/off-peak pricing are approximated (see CHANGELOG notes).
  Codex reports 0.0 (no price), Claude Code reports the SDK figure.

## Provider registry and custom providers

| Call | Effect |
|---|---|
| `LLMEngine.provider_aliases()` | Snapshot of alias → provider routing (read-only copy) |
| `LLMEngine.register_provider_alias(alias, provider)` | Exact-match routing |
| `LLMEngine.register_provider_rule(pattern, provider, kind="contains" \| "startswith")` | Substring/prefix routing (prepends) |
| `LLMEngine.set_default_provider(provider \| None)` | Fallback when nothing matches |

The top-level `PROVIDER_ALIASES` constant is a deprecated import-time
snapshot; use `LLMEngine.provider_aliases()`. Custom backends subclass
`BaseProvider` (exported at top level); capability flags drive
`lazybridge.matrix.provider_capabilities()`. Unsupported features raise
`UnsupportedFeatureError`. See `docs/guides/advanced/base-provider.md`.
