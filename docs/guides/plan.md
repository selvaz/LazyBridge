# Plan

`Plan` is where LazyBridge grows up. A plan declares a graph of steps
with typed hand-offs, named outputs, conditional routing, and optional
resume — all validated at construction time via `PlanCompileError`, so
a misspelled step name fails before the first LLM call.

The mental model: each `Step` is "run this target with this input,
optionally named, optionally producing a typed output, optionally
writing to a shared store". The `target` can be a tool name, a plain
callable, or an Agent (which may itself have tools or sub-agents). The
uniform Tool-is-Tool contract applies here too.

The killer features over `Agent.chain` are four: (a) typed hand-offs
via `output=Model`, so the next step can read fields rather than
re-parsing strings; (b) conditional routing via `output.next: Literal`,
so branches are data-driven; (c) parallel branches via
`Step(parallel=True)` + `from_parallel`; (d) checkpoint/resume via
`store + checkpoint_key`, so a crash mid-plan restarts the failed
step rather than the whole pipeline.

## Example

```python
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step
from pydantic import BaseModel
from typing import Literal

class Hits(BaseModel):
    items: list[str]
    next: Literal["rank", "empty"] = "rank"

class Ranked(BaseModel):
    top: list[str]

store = Store(db="research.sqlite")

plan = Plan(
    Step(searcher, name="search",  writes="hits",   output=Hits),
    Step(ranker,   name="rank",    task=from_prev,  output=Ranked),
    Step(writer,   name="write",   task=from_step("rank")),
    Step(apology,  name="empty"),  # reached only if Hits.next == "empty"
    store=store, checkpoint_key="research", resume=True,
    max_iterations=20,
)

Agent.from_engine(plan)("AI trends April 2026")
```

## Running the same Plan concurrently: `on_concurrent`

A `Plan` with a `checkpoint_key` locks that key: two runs that both
think they own "research" would race on the checkpoint and clobber
each other.  The default policy is **fail fast**; the alternative is
**fork** — each `.run()` writes under `f"{checkpoint_key}:{run_uid}"`
instead of the shared key, so many runs can execute at once without
collision.

```python
# What this shows: a pipeline intended to run N times in parallel
# across different inputs (a batch job, a fan-out from a queue).
# Why on_concurrent="fork": with the default "fail" policy, the second
# concurrent run would raise ConcurrentPlanRunError as soon as it saw
# the shared checkpoint. "fork" suffixes each run's checkpoint with a
# unique run_uid so their state is isolated.

from lazybridge import Agent, Plan, Step, Store

store = Store(db="batch.sqlite")

def build_plan():
    return Plan(
        Step(researcher, name="search",  writes="hits"),
        Step(writer,     name="write"),
        store=store,
        checkpoint_key="batch_job",
        on_concurrent="fork",   # each .run() claims "batch_job:<run_uid>"
        # resume=True is NOT supported with on_concurrent="fork" — fork
        # mode is deliberately stateless across invocations.
    )

# Fan out 10 concurrent runs.  Each gets its own isolated checkpoint
# and never collides with siblings.
import asyncio
plan_agent = Agent.from_engine(build_plan())
results = asyncio.run(asyncio.gather(*[
    plan_agent.run(f"topic {i}") for i in range(10)
]))
```

Use `on_concurrent="fail"` (the default) when the `checkpoint_key` is
a long-lived pipeline identity you want to resume across process
restarts.  Use `"fork"` when the key is a pipeline *shape* and each
run is an independent execution of it.

## Pitfalls

- Forgetting ``output=Model`` on a step and then expecting the next step
  to read ``.field`` — the next step will see a plain string. Declare
  types everywhere you need them.
- Cyclic ``depends`` or references to unknown step names → ``PlanCompileError``.
- ``resume=True`` without ``store=`` is a silent no-op (no checkpoint
  to read or write to). Pass both.
- A step that fails persists a ``status="failed"`` checkpoint pointing
  back at itself. Subsequent ``resume=True`` runs retry that step.
- ``on_concurrent="fork"`` + ``resume=True`` is rejected at construction —
  fork mode is stateless by design, there's nothing to resume from.

!!! note "API reference"

    Plan(
        *steps: Step,
        max_iterations: int = 100,
        store: Store | None = None,
        checkpoint_key: str | None = None,
        resume: bool = False,
    ) -> Engine
    
    Step(
        target: str | Callable | Agent,      # tool name, function, Agent
        task: Sentinel | str = from_prev,
        context: Sentinel | str | None = None,
        sources: list = (),
        writes: str | None = None,            # Store key under which payload is saved
        input: type = Any,
        output: type = str,                   # Pydantic triggers structured output + routing
        parallel: bool = False,
        name: str | None = None,
    )
    
    PlanCompileError  # raised at Agent construction if Plan is invalid
    PlanState         # checkpoint shape: plan_id, current_step, next_step, store, history, status
    StepResult        # single step record: step_name, envelope, ts
    
    Usage: Agent(engine=Plan(Step(a), Step(b)))

!!! warning "Rules & invariants"

    - ``max_iterations`` caps the total number of step executions in one
      ``run`` to guard against runaway routing loops (default 100). Raise
      it for legitimate long plans; lower it to fail fast during dev.
    - Step names are unique. ``PlanCompileError`` fires at Agent construction
      if duplicates or dangling references exist.
    - Sentinels: ``from_prev`` (previous step's output, default),
      ``from_start`` (original user task), ``from_step("name")``,
      ``from_parallel("name")``. See the sentinels page.
    - ``output=SomeModel`` activates structured output at that step. If the
      model has a ``next: Literal["a", "b", ...]`` field, the plan routes to
      the matching step on completion.
    - ``parallel=True`` marks a step as a concurrent branch; combine with
      ``from_parallel`` on the join step.
    - ``writes="key"`` stores the step's payload into ``store[key]`` if a
      Store is passed. Required for checkpoint data to survive across runs.
    - ``checkpoint_key`` + ``store`` enable state persistence after every
      step; ``resume=True`` reads the checkpoint and picks up at the next
      unrun step (failed runs restart from the failing step, not the next).

## See also

[sentinels](sentinels.md), [parallel_steps](parallel-steps.md),
[checkpoint](checkpoint.md), [verify](verify.md),
[plan_serialize](plan-serialize.md),
decision tree: [composition](../decisions/composition.md)
