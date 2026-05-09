# Composition

> **Composing agents: `chain`, `Agent.parallel`, `Plan`, or
> `tools=[...]`?**

Pick by **who decides what runs when** — pre-scripted by you,
LLM-driven, or declared with compile-time validation.

## Decision tree

```text
Linear pipeline, output of step N becomes task of N+1?
    → Agent.chain(a, b, c)
      # sugar for Agent(engine=Plan(Step(target=a, name=a.name),
      #                              Step(target=b, name=b.name), …))

Run the same task on N agents concurrently, get list[Envelope]?
    → Agent.parallel(a, b, c)
      # sugar for asyncio.gather over a/b/c, returns list[Envelope]

Let the LLM decide which sub-agent(s) to call (and when, including
in parallel)?
    → Agent(engine=LLMEngine("…"), tools=[a, b, c])
      # the engine fans out automatically when the model emits
      # multiple tool calls in one turn — no config needed

Declared workflow with typed hand-offs, routing, parallel bands,
or crash-resume?
    → Agent(engine=Plan(Step("…", output=…), Step("…", routes={…}), …),
            tools=[…])
```

## Quick reference

| Who decides the flow? | Use |
|---|---|
| You, linear and fixed | `Agent.chain(a, b, c)` |
| You, deterministic fan-out → list | `Agent.parallel(a, b, c)` |
| You, declared DAG with types / routing / resume | `Plan(Step(…), …)` |
| The LLM (which to call, when, in parallel) | `Agent(tools=[a, b, c])` |

## Notes

- **`Agent.chain` is sugar for a linear `Plan`.** Use it for
  one-liner sequential handoffs; reach for the explicit `Plan`
  the moment you can see a router or a parallel band coming.
- **`Agent.parallel` returns `list[Envelope]`, not a single
  envelope.** Feed the list to a follow-up summariser if you
  want one aggregated answer.
- **`tools=[a, b, c]` is the LLM-driven path.** When the model
  emits multiple tool calls in one turn, LazyBridge dispatches
  them concurrently via `asyncio.gather` — automatic
  parallelism, no flag.
- **All four shapes compose recursively.** A `Plan` step's
  `target` can be an agent built from `Agent.chain` or
  `Agent.parallel`; an `Agent.parallel` branch can itself contain
  a `Plan`. The unit at every level is the same `Agent`.

## See also

- [Chain](../guides/mid/chain.md) — sequential composition
  reference.
- [Parallel](../guides/mid/parallel.md) — `Agent.parallel`
  semantics; `_ParallelAgent` return type.
- [As tool](../guides/mid/as-tool.md) — when to pass an agent in
  `tools=[…]` directly vs `agent.as_tool("alias")`.
- [Plan](../guides/full/plan.md) — declared orchestration with
  compile-time DAG validation.
- [Parallelism decision](parallelism.md) — automatic vs declared.
