# Plan

## Example

```python
from lazybridge import Agent, Plan, Step, Store, from_prev, from_step
from pydantic import BaseModel
from typing import Literal

# next: Literal[...] field turns step completion into a routing decision.
# Plan routes to the step whose name matches the value of .next after each run.
class Hits(BaseModel):
    items: list[str]
    next: Literal["rank", "empty"] = "rank"  # "empty" → skip to apology step

class Ranked(BaseModel):
    top: list[str]

store = Store(db="research.sqlite")

# Plan(...) constructs the engine and validates the DAG — no LLM call yet.
plan = Plan(
    Step(searcher, name="search",  writes="hits",   output=Hits),
    Step(ranker,   name="rank",    task=from_prev,  output=Ranked),   # receives search's Envelope
    Step(writer,   name="write",   task=from_step("rank")),           # receives rank's Envelope by name
    Step(apology,  name="empty"),                                     # reached only if Hits.next == "empty"
    store=store, checkpoint_key="research", resume=True,
    max_iterations=20,
)

# Agent.from_engine wraps the plan; ("task") starts execution; .text() reads result.
result = Agent.from_engine(plan)("AI trends April 2026")
print(result.text())
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

