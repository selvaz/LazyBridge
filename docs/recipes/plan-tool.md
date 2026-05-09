# Plan tool

`make_planner` is a pre-built factory that wires up an LLM-driven
planner: it picks between a direct sub-agent call or a multi-step
plan automatically, based on the input. Use it as the orchestrator
when you want planning logic without writing it from scratch.

## Source

```python
--8<-- "examples/patterns/plan_tool.py"
```

## Walkthrough

- **`make_planner([research, math, writer])`** returns an `Agent`.
  The factory under `lazybridge.ext.planners.builder` constructs an
  inner `Plan` + dispatch logic; you supply the specialists.
- **Specialist `description=`** drives the planner's choice of
  which agent to call; precise descriptions matter as much as for
  the supervisor pattern.
- **Four query styles** exercise the planner's decision tree:
  trivial (no agent needed), single-agent (one specialist),
  multi-step plan (chained calls), parallel + synth (fan-out then
  combine).

## Variations

- Use `make_blackboard_planner` (see [Blackboard planner](blackboard-planner.md))
  for a flat to-do list shape instead of a DAG.
- For full control over the planning shape, build the `Plan`
  yourself — see [Agent builds a plan](agent-builds-plan.md) and
  [Dynamic re-planning](dynamic-replanning.md).
- The factory accepts `verbose=True` to surface planner decisions
  on stdout; pair with a `Session` for structured event logs.

## See also

- [Plan](../guides/full/plan.md) — the engine the factory wraps.
- [Blackboard planner](blackboard-planner.md) — sibling factory
  for flat task lists.
- [Agent builds a plan](agent-builds-plan.md) — the manual
  alternative when you want to write the planning logic yourself.
