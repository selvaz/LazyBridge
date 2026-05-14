# Sentinels & predicates

Sentinels declare **where a Plan step's input comes from**; the `when`
DSL declares **when a route fires**. Both are validated at Plan
construction time — a typo in `from_step("reseach")` raises
`PlanCompileError` before any LLM call.

For narrative usage see [Guides → Full → Sentinels](../guides/full/sentinels.md)
and [Guides → Full → Routing](../guides/full/routing.md).

| Symbol | Resolves to | Scope |
|---|---|---|
| `from_prev` | The previous step's payload | In-Plan only |
| `from_start` | The Plan's initial input | In-Plan only |
| `from_step("name")` | The named step's output | In-Plan only |
| `from_parallel("name")` | A single named parallel branch's output | In-Plan only |
| `from_parallel_all("name")` | All consecutive parallel siblings, labelled-text joined | In-Plan only |
| `from_memory("name")` | An agent's live conversation history | Cross-run, via `Memory` |
| `from_agent("name")` | An agent's last persisted output | Cross-run, via `Store` |
| `when` | Routing predicate DSL builder | `Step(routes={…})` |

## Plan-only sentinels

::: lazybridge.from_prev

::: lazybridge.from_start

::: lazybridge.from_step

::: lazybridge.from_parallel

::: lazybridge.from_parallel_all

## Universal sentinels

::: lazybridge.from_memory

::: lazybridge.from_agent

## Routing predicates

::: lazybridge.when
