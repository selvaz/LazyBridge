# Dynamic re-planning

A planner that emits one *round* of independent tasks at a time,
runs them in parallel, sees the results, then emits the next round.
Adaptive: each round is informed by the previous one. The
LangGraph-equivalent shape is "ReAct-on-tasks", but the planning
unit is a batch of tasks rather than a single tool call.

## Source

```python
--8<-- "examples/patterns/dynamic_planner.py"
```

## Walkthrough

- **`PlanRound` is a Pydantic schema** — the planner emits a list
  of tasks for the next round plus a `done: bool` flag. The
  outer loop dispatches all tasks concurrently via `asyncio.gather`,
  collects results, feeds them back to the planner.
- **`max_rounds`** is the safety net for bad termination logic —
  if `done=False` keeps firing forever, the loop bails. Set it
  defensively.
- **Per-task sequential flag** lets a round mix parallel and
  sequential tasks: the planner can declare that a specific task
  must wait for the others to finish before running.

## Variations

- Add a `verify=judge` on the planner agent itself to gate
  termination — the judge sees the latest round's results and
  decides whether `done=True` is justified.
- Persist round results to a `Store` so a debugger / dashboard
  can watch progress in real time.
- Replace the manual `asyncio.gather` with `Plan` parallel bands
  (`Step(parallel=True)`) if the round structure is fixed enough
  to declare up-front.

## Variations — anti-patterns

- **Pathological case**: planner emits `done=False` and an empty
  task list — the loop spins. The example file's source comment
  (line 196 in the upstream version) documents this; mitigate by
  adding a "no-tasks → final answer" branch.

## See also

- [Plan](../guides/full/plan.md) — declared alternative when the
  structure is known up front.
- [Parallel](../guides/mid/parallel.md) — `Agent.parallel` is
  the application-layer fan-out used inside each round.
- [Agent builds a plan](agent-builds-plan.md) — typed-spec
  alternative when the topology is decided once rather than
  re-emitted every round.
