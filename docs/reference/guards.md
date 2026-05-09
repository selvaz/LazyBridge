# Guards

Hard input / output filters that run before and after the engine.
Compose with `GuardChain`; build cheap deterministic guards with
`ContentGuard` and LLM-as-judge gates with `LLMGuard`. `GuardError`
is the exception type some integrations raise.

For narrative usage see [Guides → Mid → Guards](../guides/mid/guards.md).
For the soft (judge-and-retry) sibling see
[Guides → Mid → verify=](../guides/mid/verify.md).

::: lazybridge.Guard

::: lazybridge.GuardAction

::: lazybridge.ContentGuard

::: lazybridge.GuardChain

::: lazybridge.LLMGuard

::: lazybridge.GuardError
