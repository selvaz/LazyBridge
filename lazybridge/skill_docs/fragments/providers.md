## signature
Agent.from_provider(name: str, *, tier: str = "medium", **kw) -> Agent

Tier names: "top" | "expensive" | "medium" | "cheap" | "super_cheap"
Providers:  "anthropic" | "openai" | "google" | "deepseek"

## rules
Tier aliases resolve at construction time via _TIER_ALIASES on each provider.
Prices per 1M tokens (input/output). Context = max input window.

## Anthropic model tiers
| tier        | model              | ctx  | max_out | $/M in | $/M out |
|-------------|--------------------|------|---------|--------|---------|
| top         | claude-opus-4-7    | 1 M  | 128 K   | $5.00  | $25.00  |
| expensive   | claude-opus-4-6    | 1 M  | 128 K   | $5.00  | $25.00  |
| medium      | claude-sonnet-4-6  | 1 M  | 64 K    | $3.00  | $15.00  |
| cheap       | claude-haiku-4-5   | 200K | 64 K    | $1.00  | $5.00   |
| super_cheap | claude-3-haiku     | 200K | 4 K     | $0.25  | $1.25   |

Thinking: opus-4-7/4-6/sonnet-4-6 → adaptive (no budget_tokens).
          haiku-4-5 and 3.x → ThinkingConfig(budget_tokens=N) required.
          opus-4-7 does NOT accept temperature.
Native tools: WEB_SEARCH, CODE_EXECUTION, COMPUTER_USE

## OpenAI model tiers
| tier        | model          | ctx  | max_out | $/M in | $/M cached | $/M out |
|-------------|----------------|------|---------|--------|-----------|---------|
| top         | gpt-5.5-pro    | 1 M  | 128 K   | $30.00 | -         | $180.00 |
| expensive   | gpt-5.5        | 1 M  | 128 K   | $5.00  | $0.50     | $30.00  |
| medium      | gpt-5.4-mini   | 400K | 128 K   | $0.75  | $0.075    | $4.50   |
| cheap       | gpt-5.4-nano   | 400K | 128 K   | $0.20  | $0.02     | $1.25   |
| super_cheap | gpt-4o-mini    | 128K | 16 K    | $0.15  | -         | $0.60   |

Other models: gpt-5.4-pro ($30/$180), gpt-5.4 ($2.50 / $0.25 cache / $15),
              gpt-5 ($1.25/$10), gpt-4o ($2.50/$10), gpt-4.1 ($2/$8),
              gpt-4.1-mini ($0.40/$1.60), o3 ($2/$8), o4-mini ($1.10/$4.40)
Thinking: gpt-5.5/gpt-5.5-pro → reasoning_effort none|low|medium|high|xhigh (default medium).
          o-series + gpt-5.4-pro → reasoning_effort low|medium|high.
          Standard GPT models → no thinking support.
Native tools: WEB_SEARCH, CODE_EXECUTION, FILE_SEARCH, COMPUTER_USE, IMAGE_GENERATION
Cache: automatic via prompt_tokens_details.cached_tokens; cached_input rate
       applied when published (gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano).
Long-context surcharge (>272K input on gpt-5.x) NOT modeled — cost under-counts.

## Google model tiers
| tier        | model                         | ctx | max_out | $/M in | $/M out |
|-------------|-------------------------------|-----|---------|--------|---------|
| top         | gemini-3.1-pro-preview        | 1 M | 64 K    | $2.00  | $12.00  |
| expensive   | gemini-2.5-pro                | 1 M | 64 K    | $1.25  | $10.00  |
| medium      | gemini-3-flash-preview        | 1 M | 64 K    | $0.50  | $3.00   |
| cheap       | gemini-3.1-flash-lite-preview | 1 M | 64 K    | $0.25  | $1.50   |
| super_cheap | gemini-2.5-flash-lite         | 1 M | 64 K    | $0.10  | $0.40   |

Thinking: gemini-3.x → ThinkingConfig(thinking_level="low"|"medium"|"high").
          gemini-2.x → ThinkingConfig(thinking_budget=N); -1=auto.
Native tools: GOOGLE_SEARCH, WEB_SEARCH, GOOGLE_MAPS
WARNING: Google Search + structured output = 400 error; mutually exclusive.
WARNING: gemini-2.0-flash deprecated June 1 2026 — do not use in new code.

## DeepSeek model tiers
| tier                          | model               | ctx | max_out | $/M in | $/M in cached | $/M out |
|-------------------------------|---------------------|-----|---------|--------|---------------|---------|
| top / expensive               | deepseek-v4-pro     | 1 M | 384 K   | $1.74  | $0.145        | $3.48   |
| medium / cheap / super_cheap  | deepseek-v4-flash   | 1 M | 384 K   | $0.14  | $0.028        | $0.28   |

Thinking: both V4 models support ThinkingConfig → reasoning_content field.
          temperature/top_p/presence_penalty/frequency_penalty stripped in thinking mode.
          ThinkingConfig on non-V4 models raises ValueError.
Cache: automatic on repeated prefixes ≥1024 tokens; no opt-in needed.
DEPRECATED (retire 2026-07-24): deepseek-reasoner, deepseek-chat → both map to deepseek-v4-flash.
Native tools: none (function calling supported).

## example
```python
from lazybridge import Agent

# Tier-based selection — model name never appears in app code.
smart  = Agent.from_provider("anthropic", tier="top")
fast   = Agent.from_provider("openai",    tier="medium")
budget = Agent.from_provider("google",    tier="super_cheap")

# Direct model name (bypasses tier aliases).
a = Agent("gpt-5.4-mini")
b = Agent("claude-haiku-4-5")
```

## pitfalls
- DeepSeek V4 has 2 models (v4-pro, v4-flash); three tier aliases collapse onto v4-flash.
- gpt-5.5-mini / gpt-5.5-nano do NOT exist yet; medium/cheap tiers still use gpt-5.4-mini / gpt-5.4-nano.
- gpt-5-mini does NOT exist. The current OpenAI mini variant is gpt-5.4-mini.
- gemini-2.0-flash is deprecated June 1 2026; use gemini-2.5-flash-lite instead.
- Adaptive thinking (Anthropic claude-opus/sonnet 4.6+) ignores budget_tokens.
