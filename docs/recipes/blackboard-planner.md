# Blackboard planner

`make_blackboard_planner` returns an agent whose state is a flat
to-do list (`set_plan` / `get_plan` / `mark_done` tools). Simpler
than a DAG for exploratory work where the structure isn't known up
front and the LLM iterates on a checklist.

> **Canonical name**: `make_blackboard_planner` is a backward-compat
> alias for `blackboard_orchestrator_agent` (in
> `lazybridge.ext.planners`). New code can use either; both resolve
> to the same factory.

## Source

```python
--8<-- "examples/patterns/blackboard_planner.py"
```

## Walkthrough

- **`make_blackboard_planner()`** wires up an `Agent` with three
  tools (`set_plan`, `get_plan`, `mark_done`) and a `Store`-backed
  scratch space. The LLM treats the to-do list as state it can
  read and update.
- **No DAG validation** — the LLM is free to revise the plan,
  reorder tasks, or insert new ones at any point. This is what
  makes it "exploratory" relative to the typed `PlanSpec`
  pattern in [Agent builds a plan](agent-builds-plan.md).
- **Pair with `Memory`** to give the planner a running record of
  what's been tried, why a previous attempt failed, what the
  current state of the world looks like — useful for tasks that
  span many iterations.

## Variations

- Persist the blackboard via `Store(db="planner.sqlite")` so a
  long-running task can be paused and resumed. The to-do list
  survives across runs.
- Add a `verify=judge` to gate `mark_done` — useful when the
  planner tends to over-claim completion.
- Wrap as a sub-agent (`tools=[planner]`) of a higher-level
  orchestrator that dispatches between blackboard planning and
  other strategies.

## See also

- [Plan tool](plan-tool.md) — sibling factory; structured
  decision tree instead of a flat list.
- [Agent builds a plan](agent-builds-plan.md) — typed `PlanSpec`
  alternative when the structure IS known.
- [Store](../guides/mid/store.md) — backs the blackboard state.
