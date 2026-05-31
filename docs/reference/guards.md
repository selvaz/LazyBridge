# Guards

Hard input / output filters that run before and after the engine.
Compose with `GuardChain`; build cheap deterministic guards with
`ContentGuard` and LLM-as-judge gates with `LLMGuard`. `GuardError`
is the exception type some integrations raise.

A guard either **allows** the value through (optionally rewriting it
via `GuardAction.modified_text`) or **blocks** it.  Blocked input
short-circuits the run with `Envelope.error.type == "ValueError"`
or `"GuardBlocked"` for outputs; the agent call never raises.

For narrative usage see [Guides → Mid → Guards](../guides/mid/guards.md).
For the soft (judge-and-retry) sibling — which re-runs the engine
with judge feedback instead of blocking — see
[Guides → Mid → verify=](../guides/mid/verify.md).

| Symbol | Purpose | Cost |
|---|---|---|
| `Guard` | Base protocol — `acheck_input` / `acheck_output` | none |
| `GuardAction` | Verdict object returned from a check | none |
| `ContentGuard` | Deterministic regex/list-based filter | none |
| `GuardChain` | Compose multiple guards (first-fail wins) | sum of children |
| `LLMGuard` | LLM-as-judge guard | one LLM call per check |
| `DeduplicateGuard` | Removes repeated text blocks from input | none |
| `GuardError` | Exception some integrations raise on hard policy failures | n/a |

::: lazybridge.Guard

::: lazybridge.GuardAction

::: lazybridge.ContentGuard

::: lazybridge.GuardChain

::: lazybridge.LLMGuard

::: lazybridge.DeduplicateGuard

::: lazybridge.GuardError
