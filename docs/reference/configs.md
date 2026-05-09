# Runtime configs & testing

The four config dataclasses group repeated kwargs across many
`Agent(...)` constructions; precedence is **flat kwarg > config
object > documented default** (see
[Guides → Basic → Agent](../guides/basic/agent.md) pitfalls).
`MockAgent` is the deterministic test double for code that contains
an `Agent`.

## Runtime config objects

::: lazybridge.AgentRuntimeConfig

::: lazybridge.ResilienceConfig

::: lazybridge.ObservabilityConfig

::: lazybridge.CacheConfig

## Testing

::: lazybridge.MockAgent
