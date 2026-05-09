# Agent builds a plan

A planning agent emits a structured `PlanSpec` (Pydantic) that's
materialised into a live `Plan` with typed steps. Demonstrates the
"Plan as data" pattern — the LLM decides the topology, the runtime
validates it at construction (compile-time DAG checks), then
executes deterministically.

Supports a one-shot mode and a re-planning loop.

## Source

```python
--8<-- "examples/patterns/agent_builds_plan.py"
```

## Walkthrough

- **`PlanSpec` is a Pydantic schema** — the planning agent emits a
  JSON-serialisable spec (steps, sentinels, parallel flags) and a
  builder function compiles it into a real `Plan`. `PlanCompiler`
  catches dangling references / unknown targets at construction,
  before any LLM call runs.
- **Re-planning loop**: when `replan=True`, the planner sees the
  results of the previous round and emits a new `PlanSpec`. Bound
  the loop with a counter or a "done" predicate — there's no
  built-in safety net beyond `Plan(max_iterations=...)`.
- **Sentinels in the spec** (`from_prev`, `from_step("name")`,
  `from_parallel("name")`) translate from JSON strings back into
  the Python sentinel objects.

## Variations

- Drop the re-planning loop and run the materialised plan once —
  cheaper, deterministic.
- Persist the `PlanSpec` (`json.dumps(spec.model_dump())`) for
  audit / replay. See [Plan serialization](../guides/advanced/plan-serialize.md)
  for the runtime side.
- Constrain the planner's choice of targets via the schema
  (`Literal[...]` field for step `target`) so the LLM can't
  fabricate non-existent agents.

## See also

- [Plan](../guides/full/plan.md) — the runtime side.
- [Plan serialization](../guides/advanced/plan-serialize.md) —
  for persisting the spec across processes.
- [Dynamic re-planning](dynamic-replanning.md) — sibling pattern
  using a flat round-of-tasks shape.
- [Plan tool](plan-tool.md) — the pre-built factory alternative.
