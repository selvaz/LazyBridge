# Checkpoint & resume

## Example

```python
from lazybridge import Agent, Plan, Step, Store

store = Store(db="pipeline.sqlite")

def build_plan():
    return Plan(
        Step(researcher, name="search",  writes="hits"),
        Step(ranker,     name="rank",    writes="ranked"),
        Step(writer,     name="write",   writes="draft"),
        store=store,
        checkpoint_key="pipeline",
        resume=True,
    )

# Run 1 — crashes after rank: status="failed", next_step="write".
try:
    Agent.from_engine(build_plan())("AI trends")
except KeyboardInterrupt:
    pass

# Run 2 — resumes from the failing step; search+rank are not re-run.
Agent.from_engine(build_plan())("AI trends")

# Run 3 — plan is already "done": short-circuits, returns cached kv.
result = Agent.from_engine(build_plan())("AI trends")
print(result.payload)  # {"hits": ..., "ranked": ..., "draft": ...}
```

## Pitfalls

- Changing the Plan definition (adding/removing/renaming steps) and
  resuming from an old checkpoint will fail: the saved ``next_step``
  may no longer exist. Delete the checkpoint
  (``store.delete(checkpoint_key)``) after refactoring steps.
- Non-JSON-serialisable ``writes`` values (e.g. a file handle) are
  stringified silently via ``default=str``. Prefer primitives and
  Pydantic models.
- Resume does not re-inject the original session or exporters; pass the
  same ``session=`` + ``store=`` on every run for continuity.

!!! note "API reference"

    Plan(
        *steps,
        store: Store,
        checkpoint_key: str,
        resume: bool = False,
    ) -> Engine
    
    # Persisted shape at store[checkpoint_key]:
    #   {
    #     "next_step": str | None,
    #     "kv": {"writes_key": payload, ...},
    #     "completed_steps": [str],
    #     "status": "running" | "failed" | "done",
    #   }

!!! warning "Rules & invariants"

    - Checkpoint fires after each successful step and after each failed step.
    - Success path: ``status="running"`` (next step pending) →
      ``status="done"`` when ``next_step is None``.
    - Fail path: the failing step is NOT added to ``completed_steps``;
      the checkpoint saves ``next_step=<failing step name>`` +
      ``status="failed"``. A subsequent run with ``resume=True`` restarts
      from that step.
    - Success + ``resume=True`` + ``status="done"`` → short-circuit: Plan
      returns an Envelope with payload = cached ``kv``, without re-running.
    - Checkpoint is JSON-encoded via ``Store.write``; ``writes=`` payloads
      must be JSON-serialisable (string, dict, Pydantic model via
      ``.model_dump()``).

