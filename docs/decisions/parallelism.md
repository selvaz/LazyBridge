# Parallelism

> **Automatic or declared?**

There's no serial / parallel mode switch. Automatic parallelism is
always on when the model emits multiple tool calls; declared
parallelism is when you fix the shape yourself.

## Decision tree

```text
Want the LLM to decide whether to run things in parallel?
    → Pass them in tools=[...] on a regular Agent. When the model
      emits multiple tool calls in one turn, the engine runs them
      concurrently via asyncio.gather. No config.

Want to declare that N agents run at once on the same task?
    → Agent.parallel(a, b, c)(task)            # → list[Envelope]

Want declared concurrent branches inside a typed workflow?
    → Plan(
          Step(a, parallel=True),
          Step(b, parallel=True),
          Step(join,
               task="Aggregate the branches.",
               context=[from_parallel("a"), from_parallel("b")]),
      )
```

## Quick reference

| Who decides the parallelism shape? | Use |
|---|---|
| The LLM (emergent, per turn) | `Agent(tools=[a, b, c])` |
| You (deterministic fan-out → list) | `Agent.parallel(a, b, c)` |
| You (typed workflow with bands) | `Plan(Step(…, parallel=True), …)` |

## Notes

- **Automatic parallelism is the default.** When the engine sees
  multiple `tool_call` messages in one model response, it
  dispatches them concurrently. There is no setting that turns
  this off.
- **`Agent.parallel(...)` is scripted fan-out.** Every input
  agent runs unconditionally on the same task; the result is
  `list[Envelope]` in input order. Use this when you *know* you
  want N things to happen.
- **`Step(parallel=True)` bands are atomic.** If any branch
  errors, no `writes=` from the band are applied — a future
  `resume=True` re-runs the whole band cleanly. This is why
  cross-branch side effects (Store writes, external POSTs) need
  idempotency keys.
- **Only consecutive `parallel=True` steps form a band.** A
  non-parallel step in between starts a new band; keep parallel
  siblings contiguous in the declaration.

## See also

- [Parallel](../guides/mid/parallel.md) — `Agent.parallel`
  reference (returns `_ParallelAgent`, not `Agent`).
- [Parallel plan steps](../guides/full/parallel-plan-steps.md)
  — band semantics, `from_parallel_all`, atomicity rules.
- [Composition decision](composition.md) — which composition
  shape to pick.
