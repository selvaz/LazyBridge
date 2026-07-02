# Runtime configs & testing

The 0.7-era ``AgentRuntimeConfig`` / ``ResilienceConfig`` /
``ObservabilityConfig`` wrapper-of-flat-kwargs configs were deleted in
0.7.9 — they bundled flat kwargs into shareable objects with a
``flat kwarg > config object > default`` precedence game that required
a private ``_UNSET`` sentinel on every kwarg (a documented LLM trap,
T14 in the audit).

For fleet management, use a Python dict spread:

```python
PROD_DEFAULTS = dict(
    timeout=60,
    max_retries=5,
    max_output_retries=2,
    cache=True,
    verbose=False,
    session=session,
)

researcher = Agent(**PROD_DEFAULTS, engine=LLMEngine("model"), name="research")
writer     = Agent(**PROD_DEFAULTS, engine=LLMEngine("model"), name="write")
```

Same end-user value, no precedence-game complexity, no sentinel.

## Cache config

``CacheConfig`` carries real semantic value (``enabled``, ``ttl``)
consumed inside ``LLMEngine``.  As of the v1 API pass it lives under
``lazybridge.core.types`` (import it from there); the top-level
re-export is deprecated (0.10) and will be removed in 1.0.

::: lazybridge.core.types.CacheConfig

## Testing

::: lazybridge.MockAgent
