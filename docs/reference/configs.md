# Runtime configs & testing

The 0.7-era ``AgentRuntimeConfig`` / ``ResilienceConfig`` /
``ObservabilityConfig`` wrapper-of-flat-kwargs configs were deleted in
0.8.0 — they bundled flat kwargs into shareable objects with a
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

## Cache config (kept)

``CacheConfig`` is intentionally kept — it carries real semantic
value (``enabled``, ``ttl``) consumed inside ``LLMEngine``.

::: lazybridge.CacheConfig

## Testing

::: lazybridge.MockAgent
