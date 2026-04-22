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

## Typed hand-offs: `input=` and `output=`

The killer feature of Plan over `Agent.chain` is **typed hand-offs
validated at construction time**.  Declare each step's input and
output with a Pydantic class and the `PlanCompiler` ensures, before a
single token is spent, that step N's output matches step N+1's
expected input.

```python
# What this shows: two steps with a typed hand-off. The ranker
# declares input=Hits so the compiler checks that the preceding
# step's output= is also Hits (or Any). A typo or refactor that
# breaks the contract surfaces at Agent(engine=plan) construction,
# not two minutes into a production run.
# Why: string-threaded pipelines silently carry type drift; a
# Pydantic-threaded pipeline cannot.

from lazybridge import Agent, Plan, Step
from pydantic import BaseModel

class Hits(BaseModel):
    items: list[str]

class Ranked(BaseModel):
    top: list[str]

plan = Plan(
    Step(searcher, name="search", output=Hits),    # produces Hits
    Step(ranker,   name="rank",
         input=Hits,                               # compile-time check:
                                                   #   prev step's output must be Hits
         output=Ranked),
    Step(writer,   name="write", input=Ranked),
)

# If ``searcher`` returned ``output=SomethingElse``, this line would
# raise PlanCompileError at construction:
Agent.from_engine(plan)("AI trends")
```

`input=Any` (the default) disables the check for that step.  Use it
when the step accepts multiple shapes (a summariser that works on any
string) or when you're deliberately coercing at the boundary.

## Conditional routing: `output.next: Literal[...]`

When a step's output model carries a `next` field typed as
`Literal[...]`, the Plan engine uses its runtime value as the name of
the **next step to run** — skipping over intermediate steps.  This is
how you declare branches without writing control flow.

```python
# What this shows: a branch that skips the writer and routes to an
# "empty" apology step when the search returned nothing.
# Why declarative: the branch is part of the DATA, not a Python
# conditional — the compiler validates that every value in
# Literal["rank", "empty"] is a real step name, and the graph can be
# serialised to JSON/YAML without losing the branch shape.

from typing import Literal
from lazybridge import Plan, Step
from pydantic import BaseModel

class Hits(BaseModel):
    items: list[str]
    # Runtime value of ``next`` determines which step runs after this one.
    # PlanCompiler checks that "rank" and "empty" are both known step
    # names; a typo ("writ" vs "write") fails at construction.
    next: Literal["rank", "empty"] = "rank"

plan = Plan(
    Step(searcher, name="search", output=Hits),      # next field drives routing
    Step(ranker,   name="rank",   output=Ranked),    # default path
    Step(writer,   name="write",  task=from_step("rank")),
    Step(apology,  name="empty"),                    # branch target: Hits.next == "empty"
)
```

`max_iterations` (default 100) is the loop-budget safety valve: if
routing bounces between branches without ever hitting the end, the
plan fails with a clear error rather than running forever.

## Step-level `sources=` and `context=`

Alongside `task=` (sentinel or literal), each Step accepts
`context=` (additional context — sentinel or string) and `sources=`
(live-view objects with `.text()`).  These compose with whatever
Agent-level sources the step's target already carries.

```python
# What this shows: feeding a step both a dynamic context from a
# previous step AND live state from a shared Store, without touching
# the step agent's own configuration.
# Why step-level: keeps the target Agent reusable across plans;
# one Agent can be a step in many pipelines with different
# context/sources on each.

from lazybridge import Plan, Step, Store, from_step

store = Store(db="shared.sqlite")

plan = Plan(
    Step(fetcher,   name="fetch", writes="hits"),
    Step(summariser, name="sum",
         task=from_step("fetch"),          # input data
         context=from_step("fetch"),       # same envelope as context too —
                                           #   "text says X; in the context of X"
         sources=[store]),                 # live store content injected each run
)
```

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
