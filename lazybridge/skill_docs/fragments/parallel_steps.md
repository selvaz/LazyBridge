## signature
Step(target, *, parallel: bool = False, name: str | None = None, ...)
from_parallel(name: str) -> Sentinel

# Typical shape: N parallel branches followed by a join step.
Plan(
    Step(a, name="a", parallel=True),
    Step(b, name="b", parallel=True),
    Step(c, name="c", parallel=True),
    Step(join, name="join",
         task=from_parallel("a"),
         context=from_parallel("b")),
)

## rules
- ``parallel=True`` marks a step as a branch that runs concurrently
  with other consecutive parallel steps in the plan.
- The plan engine dispatches all consecutive ``parallel=True`` steps
  via ``asyncio.gather`` before proceeding.
- A non-parallel step immediately after parallel steps acts as an
  implicit join: it sees ``from_prev`` as the last completed branch's
  output; use ``from_parallel("name")`` to reach a specific branch.
- Parallel steps may have their own ``writes=`` — each branch's
  payload is persisted under the respective Store key.
- Errors in a parallel branch surface as an error ``Envelope`` for
  that branch only; sibling branches continue.

## narrative
**Use `parallel=True` step bands** when independent steps can run
concurrently and the next step needs all of their results.
`from_parallel_all("first")` aggregates the band's outputs as a labelled
text join.  Atomicity: if any branch errors, no `writes` are applied —
a future resume re-runs the whole band cleanly.

**Use `Agent.parallel` instead** for simple deterministic fan-out at
the application layer (no Plan, no aggregation, just `list[Envelope]`).

## example
```python
from lazybridge import Agent, Plan, Step, from_parallel, Store

store = Store(db="monitor.sqlite")

plan = Plan(
    # Three independent searchers fan out in parallel.
    Step(anthropic_search, name="search_a", parallel=True, writes="findings_a"),
    Step(openai_search,    name="search_o", parallel=True, writes="findings_o"),
    Step(google_search,    name="search_g", parallel=True, writes="findings_g"),

    # Join: synthesiser reads all three branches via context=.
    Step(synthesiser, name="synth",
         task=from_parallel("search_a"),
         context=from_parallel("search_o"),  # could concatenate more
         writes="plan"),

    store=store,
)
Agent.from_engine(plan)("framework update — April 2026")
```

## pitfalls
- Interleaving parallel and sequential steps without care: the engine
  only bundles CONSECUTIVE ``parallel=True`` steps. Insert them in a
  run.
- Forgetting the join step — after N parallel steps the next
  non-parallel step IS the join. If you want all three outputs you
  must read them via ``from_parallel("…")`` on the join step;
  otherwise only ``from_prev`` (last completed) is visible.
- Checkpointing across a parallel block is coarse-grained: the engine
  saves after the block completes, not per-branch. If branch A
  succeeds but B crashes, resume retries the whole block, not just B.
  (Tracked for future work.)

## see-also
- [Plan](plan.md) — the engine that orchestrates parallel bands.
- [Sentinels](sentinels.md) — `from_parallel_all` and `from_parallel`.
