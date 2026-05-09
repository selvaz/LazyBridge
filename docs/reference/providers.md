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
| `LLMEngine.set_default_provider(provider | None)` | Fallback when no rule matches; `None` disables the safety net |
