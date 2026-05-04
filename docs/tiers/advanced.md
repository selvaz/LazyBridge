# Advanced tier

**Use this when** you are extending the framework itself: adding a new LLM provider, writing a custom execution engine, or serialising Plans across processes.

**Skip this tier** if you're building apps — Basic/Mid/Full cover everything user-facing.

## Walkthrough

Advanced is for framework extension.  The `Engine` protocol is
one method (`run`) and one optional one (`stream`); implement it
to add a new execution model alongside `LLMEngine` /
`Plan` / `HumanEngine` / `SupervisorEngine`.

`BaseProvider` is the contract every LLM provider satisfies —
four methods, full retry/backoff handled for you by `Executor`.
The `Provider registry` is the runtime entry point: register
aliases / rules so `Agent("my-model")` routes to the new
provider without forking the framework.  The `LiteLLM bridge`
is the catch-all alternative — unlock 100+ extra providers via
the `litellm/<model>` prefix without writing a provider class.

`Plan.to_dict` / `from_dict` round-trip a Plan's topology
through JSON for cross-process re-use; callables and Agents are
rebound via `registry={name: target}`.  `core.types` carries the
types that flow across all these boundaries.

## Topics

* [Engine protocol](../guides/engine-protocol.md)
* [BaseProvider](../guides/base-provider.md)
* [Plan serialization](../guides/plan-serialize.md)
* [Provider registry](../guides/register-provider.md)
* [core.types](../guides/core-types.md)

## Also see

* [LiteLLM bridge](../guides/litellm.md) — 100+ extra providers via `litellm/<model>`
* [Core vs Extension policy](../guides/core-vs-ext.md) — pre-1.0 alpha posture and the import boundary

## Next steps

* [Engine protocol guide →](../guides/engine-protocol.md)
* [BaseProvider guide →](../guides/base-provider.md)
* [LiteLLM bridge →](../guides/litellm.md) — unlock 100+ extra providers via the `litellm/` prefix
* [Core vs Extension policy →](../guides/core-vs-ext.md) — pre-1.0 alpha posture and the import boundary

**By the end of Advanced you can:**

- implement a brand-new LLM provider against `BaseProvider`
- implement a brand-new execution engine against the `Engine` protocol
- serialise a `Plan` to JSON for cross-process re-use
- register custom model-string aliases at runtime
