# Custom providers

`BaseProvider` is the stable extension point for integrating any LLM
backend. The provider registry on `LLMEngine` routes model strings
to registered providers.

For narrative usage see
[Guides → Advanced → BaseProvider](../guides/advanced/base-provider.md)
and [Guides → Advanced → Providers](../guides/advanced/providers.md)
(built-in catalogue + tier tables).

## Abstract base class

::: lazybridge.BaseProvider

## Provider registry surface

The registry methods are class-level on `LLMEngine`. They mutate
class-level tables (`_PROVIDER_ALIASES`, `_PROVIDER_RULES`,
`_PROVIDER_DEFAULT`) and are documented under the engine class itself
— see [Engines → LLMEngine](engines.md#lazybridge.LLMEngine) for the
full method list. Quick reference:

| Method | Effect |
|---|---|
| `LLMEngine.register_provider_alias(alias, provider)` | Exact-match (case-insensitive) routing |
| `LLMEngine.register_provider_rule(pattern, provider, *, kind="contains" | "startswith")` | Substring / prefix routing; new rules **prepend** the rule list |
| `LLMEngine.set_default_provider(provider | None)` | Fallback when no rule matches; `None` (the 0.7.9 default) makes unknown-model strings raise ``ValueError`` instead of silently routing to Anthropic |

## stop_reason normalisation

Each provider exposes its own raw finish-reason vocabulary; LazyBridge
maps them to a normalised ``CompletionResponse.stop_reason`` so engine
loops can decide identically across providers.  Notable mappings:

| Provider | Raw value | Normalised |
|---|---|---|
| Anthropic | ``end_turn`` / ``tool_use`` / ``max_tokens`` / ``stop_sequence`` | ``end_turn`` / ``tool_use`` / ``max_tokens`` / ``end_turn`` |
| OpenAI | ``stop`` / ``tool_calls`` / ``length`` / ``content_filter`` | ``end_turn`` / ``tool_use`` / ``max_tokens`` / ``error`` |
| Google | ``STOP`` / ``MAX_TOKENS`` / ``SAFETY`` / ``RECITATION`` / ``BLOCKLIST`` | ``end_turn`` / ``max_tokens`` / ``error`` (the bucket for non-stop terminations) |
| DeepSeek | passes through OpenAI shape | as OpenAI |

The Google ``MAX_TOKENS`` mapping is fixed in 0.7.9 — pre-fix it was
returned as the literal string and broke loops that branched on
``stop_reason == "max_tokens"``.  Inspect ``Envelope.metadata.stop_reason``
to read the normalised value.
