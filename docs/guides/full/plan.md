# Plan

The deterministic-orchestration engine. A `Plan` is a declared
sequence of `Step`s with explicit data flow, validated at construction
time — broken references, duplicate names, and unknown route targets
fail before any LLM call. Pass it to an `Agent` like any other engine.

## Signature

```python
from lazybridge import Agent, Plan, Step, Store

Plan(
    *steps,                        # one or more Step instances
    max_iterations=100,            # cap on total step executions per run; guards against routing loops
    store=None,                    # Store for writes= / checkpointing
    checkpoint_key=None,           # str — required for resume
    resume=False,                  # pick up at the failed step (Phase 3b: Checkpoint & resume)
    on_concurrent="fail",          # "fail" | "fork"
)



# Concurrent fan-out — N inputs against the same Plan shape.
# Pair with on_concurrent="fork" so each input run claims its own
# isolated keyspace.
plan.run_many(tasks, *, concurrency=None, ...)   # sync — returns list[Envelope]
await plan.arun_many(tasks, *, concurrency=None, ...)  # async equivalent


# Construction errors (all raised at Agent construction).
PlanCompileError                   # invalid DAG: dangling refs, duplicates, malformed routes
ConcurrentPlanRunError             # raised at runtime CAS when two runs share a checkpoint_key

# Persisted state shapes.
PlanState                          # checkpoint: plan_id, current_step, next_step, store, history, status
StepResult                         # one record per executed step: step_name, envelope, ts


# Use as an Agent engine.
pipeline = Agent(
    engine=Plan(Step("a"), Step("b")),
    tools=[a, b],
)
```

## Synopsis

`Plan` is the engine for declared, multi-step pipelines. Every step
has a named target, a typed input/output, an explicit data source
(via [sentinels](sentinels.md)), and optionally writes its payload
to a [`Store`](../mid/store.md) bucket the rest of the pipeline can
read. All of that is validated at **construction time** —
`PlanCompileError` fires before the agent ever runs.

A `Plan` does not call an LLM by itself; it dispatches each step to
its `target` (an `Agent`, a callable, or a tool name resolved on the
wrapping agent). The orchestration layer is the deterministic part;
the per-step targets are where LLMs (or other engines) actually run.

`Agent.chain(*agents)` is the one-line sugar for the simplest `Plan`
shape — purely linear text hand-offs. Reach for `Plan` directly when
you need any of:

- typed hand-offs (`Step(output=Model)` instead of free-form text);
- conditional routing (`Step(routes=...)` or `Step(routes_by="field")`);
- parallel bands (`Step(parallel=True)`);
- named writes to a `Store`;
- crash resume.

## When to use it

- **Multi-step pipelines that need to be auditable.** The DAG is
  visible at construction; reviewers can read a `Plan(Step(...),
  Step(...))` block and know the topology without running anything.
- **Production workflows where the LLM should *not* decide the
  order.** When determinism, repeatability, or cost predictability
  matter, lift control flow out of the model and into the `Plan`.
- **Pipelines that span multiple agents and need typed payloads
  between them.** `Step(output=Model)` preserves the type at the
  step boundary; `Agent.chain` flattens to text.
- **Workflows with conditional branching, fan-out / fan-in,
  early-out, or self-correction loops.** `routes`, `routes_by`,
  `parallel=True`, and `from_parallel_all` cover the canonical
  shapes.
- **Crash-resumable runs.** `Plan(store=..., checkpoint_key=...,
  resume=True)` writes plan state after every step; a re-run with
  `resume=True` picks up at the failed step.

## When NOT to use it

- **One agent, one model call.** That's `Agent(engine=LLMEngine(...))`.
  No `Plan` needed.
- **Linear text hand-offs with no other features.** Use
  `Agent.chain(...)` — it's sugar for the simplest `Plan` and
  reads better at the call site.
- **LLM-directed dispatch** ("the model decides which agent to
  call"). Use `Agent(tools=[a, b, c])`. `Plan` is for the
  *opposite* case — explicit, declared flow.
- **Deterministic fan-out → list of envelopes.** Use
  [`Agent.parallel(...)`](../mid/parallel.md) — its return shape
  is `list[Envelope]`, which a `Plan` step can't natively produce.

## Example

```python
from pydantic import BaseModel

from lazybridge import Agent, LLMEngine, Plan, Step, Store, from_prev, from_step


class Hits(BaseModel):
    items: list[str]


class Ranked(BaseModel):
    top: list[str]


def search_web(query: str) -> str:
    """Return search hits for ``query``."""
    return "..."


searcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    tools=[search_web],
    name="search",
)
ranker = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="rank",
)
writer = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="write",
)


# 1) Linear typed pipeline — search → rank → write.
pipeline = Agent(
    engine=Plan(
        Step("search",
             task="Search the web for the user's topic.",
             writes="hits",
             output=Hits),
        Step("rank",
             task="Rank these search hits by relevance; return the top 5.",
             context=from_prev,
             output=Ranked),
        Step("write",
             task="Write a 200-word brief from the ranked items below.",
             context=from_step("rank")),
    ),
    tools=[searcher, ranker, writer],
    store=Store(db="research.sqlite"),
)
result = pipeline("AI trends April 2026")
print(result.text())


# 2) The same pipeline as Agent.chain (sugar — only works because the
#    flow is purely linear with text hand-offs).
sugar_pipeline = Agent.chain(searcher, ranker, writer, name="research")
```

For the full surface — typed payloads, routing, parallel bands,
crash-resume, fan-out runs — see the dedicated guides:
[Step](step.md), [Sentinels](sentinels.md),
[Routing](routing.md), and the Phase 3b guides
*Parallel plan steps* and *Checkpoint & resume*.

## Pitfalls

- **`max_iterations` is a safety net for routing loops** (default
  100). Hitting the cap returns a `MaxIterationsExceeded` error
  envelope — not a crash. Lower it during development to fail
  fast; raise it for legitimate long plans.
- **Cyclic routing is not a compile error.** `routes` cycles
  (`A → B → A`) may be intentional (self-correction loops) and
  surface at runtime as `MaxIterationsExceeded`. Pair every
  loop-routing pattern with a counter or termination predicate.
- **`resume=True` without `store=` is a silent no-op.** Pass both,
  and pick a `checkpoint_key`.
- **`on_concurrent="fork"` + `resume=True` is a configuration
  error.** Fork mode gives each run its own keyspace, so there's
  no shared checkpoint to resume from. The framework raises at
  construction.
- **`PlanCompileError` catches** duplicate step names, dangling
  `from_step` / `from_parallel` / `from_parallel_all` references,
  forward references, mid-band `from_parallel_all` start, unknown
  `routes=` targets, malformed `routes_by=` Literal types, and
  predicates that aren't callable. Read the error message — it
  names the offending step.
- **Plan writes go through the same `Store` as application
  writes.** Namespace your keys (`"pipeline_research/hits"` rather
  than `"hits"`) so a step's `writes=` doesn't collide with
  unrelated state.
- **`Step("name")` resolves the name on the wrapping agent's
  `tools=[...]` map.** `Plan(Step("research"))` with no
  `tools=[...]` on the agent is a `PlanCompileError` — the target
  has nowhere to resolve. `Step(target=researcher)` (the agent
  itself) is the alternative — it dispatches via `target.run()`
  directly with no tool-map lookup.

## See also

- [Step](step.md) — the per-step anatomy: target, task, context,
  sources, writes, output.
- [Sentinels](sentinels.md) — wiring data between steps
  (`from_prev`, `from_step`, `from_parallel_all`, `from_memory`,
  `from_agent`).
- [Routing](routing.md) — `routes={...}` predicate map and
  `routes_by="field"` LLM-decided dispatch, plus `when` DSL.
- [Chain](../mid/chain.md) — the sugar for the linear case.
- *Guides → Full → Parallel plan steps* (Phase 3b) — `parallel=True`
  bands and `from_parallel_all` aggregation.
- *Guides → Full → Checkpoint & resume* (Phase 3b) — `store=`,
  `checkpoint_key=`, `resume=True`, `on_concurrent=`.
- [Canonical vs sugar](../../concepts/canonical-vs-sugar.md) —
  `Agent(engine=Plan(*steps))` vs `Agent(engine=Plan(*steps))`.
