# Providers & model tiers

LazyBridge routes `Agent("model-name")` to a provider automatically. You can
also address providers by **tier name** so your code never hard-codes a model:

```python
agent = Agent.from_provider("anthropic", tier="medium")   # claude-sonnet-4-6
agent = Agent.from_provider("openai",    tier="expensive") # gpt-5.5
agent = Agent.from_provider("google",    tier="top")      # gemini-3.1-pro-preview
```

Prices below are per 1M tokens (input / output). **Context** = maximum input
window. **Max out** = maximum output tokens per call. Prices are approximate —
verify at each provider's pricing page before billing decisions.

---

## Anthropic (Claude)

| Tier | Model | Context | Max out | $/M in | $/M out |
|------|-------|--------:|-------:|-------:|--------:|
| `top` | claude-opus-4-7 | 1 M | 128 K | $5.00 | $25.00 |
| `expensive` | claude-opus-4-6 | 1 M | 128 K | $5.00 | $25.00 |
| `medium` | claude-sonnet-4-6 | 1 M | 64 K | $3.00 | $15.00 |
| `cheap` | claude-haiku-4-5 | 200 K | 64 K | $1.00 | $5.00 |
| `super_cheap` | claude-3-haiku | 200 K | 4 K | $0.25 | $1.25 |

**Thinking / reasoning**

| Models | Config |
|--------|--------|
| claude-opus-4-7, claude-opus-4-6, claude-sonnet-4-6 | `ThinkingConfig()` → `{"type": "adaptive"}` — effort controlled by the model; no `budget_tokens` |
| claude-haiku-4-5, claude-sonnet-4-5, claude-3.x | `ThinkingConfig(budget_tokens=N)` → explicit token budget required |

!!! note
    claude-opus-4-7 does **not** accept `temperature`, `top_p`, or `top_k` — passing
    them emits a `UserWarning` and they are silently dropped.

**Native tools:** `WEB_SEARCH` · `CODE_EXECUTION` · `COMPUTER_USE`

---

## OpenAI

| Tier | Model | Context | Max out | $/M in | $/M in (cached) | $/M out |
|------|-------|--------:|-------:|-------:|----------------:|--------:|
| `top` | gpt-5.5-pro | 1 M | 128 K | $30.00 | — | $180.00 |
| `expensive` | gpt-5.5 | 1 M | 128 K | $5.00 | $0.50 | $30.00 |
| `medium` | gpt-5.4-mini | 400 K | 128 K | $0.75 | $0.075 | $4.50 |
| `cheap` | gpt-5.4-nano | 400 K | 128 K | $0.20 | $0.02 | $1.25 |
| `super_cheap` | gpt-4o-mini | 128 K | 16 K | $0.15 | — | $0.60 |

GPT-5.5 (released 2026-04-23) is the new flagship. There is no `gpt-5.5-mini` /
`gpt-5.5-nano` yet, so `medium` / `cheap` continue to point at the GPT-5.4 family.

**Other available models** (not in tier aliases, still in price table):

| Model | Context | $/M in | $/M in (cached) | $/M out | Notes |
|-------|--------:|-------:|----------------:|--------:|-------|
| gpt-5.4-pro | 1 M | 128 K | $30.00 | — | $180.00 — extended reasoning flagship |
| gpt-5.4 | 1 M | 128 K | $2.50 | $0.25 | $15.00 — prior flagship |
| gpt-5 | 400 K | $1.25 | — | $10.00 | strong general model |
| gpt-4o | 128 K | $2.50 | — | $10.00 | stable legacy workhorse |
| gpt-4.1 | 1 M | $2.00 | — | $8.00 | |
| gpt-4.1-mini | 1 M | $0.40 | — | $1.60 | |
| o3 | 200 K | $2.00 | — | $8.00 | reasoning model |
| o4-mini | 200 K | $1.10 | — | $4.40 | reasoning model |

**Thinking / reasoning**

| Models | Config |
|--------|--------|
| gpt-5.5, gpt-5.5-pro | `ThinkingConfig(effort="none"\|"low"\|"medium"\|"high"\|"xhigh")`. `none` skips reasoning; default is `medium`. |
| o3, o4-mini, o1, o1-pro | `ThinkingConfig(effort="low"\|"medium"\|"high")` → `reasoning_effort` parameter; no `temperature` |
| gpt-5.4 family | `reasoning_effort` parameter; gpt-5.4-pro is reasoning-by-default |
| All others | `ThinkingConfig` has no effect; use `temperature` for creativity |

**Native tools:** `WEB_SEARCH` · `CODE_EXECUTION` · `FILE_SEARCH` · `COMPUTER_USE`

!!! note "Long-context surcharge"
    Prompts above **272K input tokens** on GPT-5.5 / GPT-5.4 are billed at 2× input
    and 1.5× output for the session. The framework's reported `cost_usd` does
    **not** apply this surcharge — it under-counts in that regime.

!!! note "Cache hits"
    OpenAI applies prompt caching automatically; the API returns
    `prompt_tokens_details.cached_tokens` (Chat Completions) /
    `input_tokens_details.cached_tokens` (Responses), which the framework
    consumes to bill cached tokens at the cached-input rate when one is
    published for the model.

---

## Google (Gemini)

| Tier | Model | Context | Max out | $/M in | $/M out |
|------|-------|--------:|-------:|-------:|--------:|
| `top` | gemini-3.1-pro-preview | 1 M | 64 K | $2.00 | $12.00 |
| `expensive` | gemini-2.5-pro | 1 M | 64 K | $1.25 | $10.00 |
| `medium` | gemini-3-flash-preview | 1 M | 64 K | $0.50 | $3.00 |
| `cheap` | gemini-3.1-flash-lite-preview | 1 M | 64 K | $0.25 | $1.50 |
| `super_cheap` | gemini-2.5-flash-lite | 1 M | 64 K | $0.10 | $0.40 |

**Thinking / reasoning**

| Models | Config |
|--------|--------|
| gemini-3.x | `ThinkingConfig(effort="low"|"medium"|"high")` → `ThinkingConfig(thinking_level=...)` |
| gemini-2.x | `ThinkingConfig(budget_tokens=N)` → `ThinkingConfig(thinking_budget=N)`; `-1` = auto |

**Native tools:** `GOOGLE_SEARCH` · `WEB_SEARCH` (alias for Google Search) · `GOOGLE_MAPS`

!!! warning "Incompatibility"
    Google Search grounding and structured output (`output=SomeModel`) cannot be
    combined in a single call — the Gemini API returns 400.

!!! warning "Deprecation"
    `gemini-2.0-flash` is deprecated and removed from tier aliases. Kept in the
    price table for compatibility only. Do not use in new code.

---

## DeepSeek

| Tier | Model | Context | Max out | $/M in | $/M in (cached) | $/M out |
|------|-------|--------:|-------:|-------:|----------------:|--------:|
| `top` / `expensive` | deepseek-v4-pro | 1 M | 384 K | $1.74 | $0.145 | $3.48 |
| `medium` / `cheap` / `super_cheap` | deepseek-v4-flash | 1 M | 384 K | $0.14 | $0.028 | $0.28 |

Both V4 models share the same tier family; three tiers collapse onto `deepseek-v4-flash`.
Cache hits apply automatically to any repeated prefix ≥ 1,024 tokens; no opt-in required.

**Thinking / reasoning**

Both `deepseek-v4-pro` and `deepseek-v4-flash` support optional thinking mode,
activated by passing `ThinkingConfig` to the request. Chain-of-thought surfaces
in the `reasoning_content` field in both streaming and non-streaming responses.

In thinking mode the API silently ignores `temperature`, `top_p`,
`presence_penalty`, and `frequency_penalty`; `tool_choice` is also not supported.

!!! note
    Passing `ThinkingConfig` to a non-V4 DeepSeek model raises `ValueError`.

!!! warning "Deprecation"
    `deepseek-reasoner` and `deepseek-chat` are deprecated and will be retired
    **2026-07-24**. They currently map to `deepseek-v4-flash` on the API side.
    Migrate to `deepseek-v4-pro` / `deepseek-v4-flash` before that date.

**Native tools:** None (standard function calling supported).

---

## Pricing references

Prices are approximate and change frequently. Always verify before billing:

- Anthropic — [console.anthropic.com/pricing](https://console.anthropic.com/pricing)
- OpenAI — [platform.openai.com/docs/pricing](https://platform.openai.com/docs/pricing)
- Google — [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing)
- DeepSeek — [platform.deepseek.com/api-docs/pricing](https://platform.deepseek.com/api-docs/pricing)
