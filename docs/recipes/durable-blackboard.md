# Durable blackboard

`durable_blackboard_agent` is the blackboard planner for agents that stay up:
the to-do list lives in a `Store` instead of a closure, so it survives the run,
the process, and the crash.

Reach for it when an agent wakes on a schedule, does one thing, and has to know
where it is next time. For a one-shot planner inside a single call, the
[flat blackboard](blackboard-planner.md) is simpler — it resets on every
invocation, which is exactly what this one must not do.

## Source

```python
--8<-- "examples/patterns/durable_blackboard.py"
```

## The five verbs

| Tool | What it does |
|---|---|
| `set_plan(reasoning, tasks)` | Creates the plan. **Refuses** to discard one still in progress. |
| `get_plan()` | The whole board: what is done, claimed, failed, and what is next. |
| `claim_next()` | Takes exactly one task, atomically. |
| `mark_done(index, summary)` | Closes a task with its result. |
| `mark_failed(index, error)` | Hands a task back after a genuine failure. |

Closing requires an **active claim**: a task that was never claimed, or one
already closed, is refused. Each planner instance also claims under its own
identity, so a run whose lease expired cannot come back and overwrite the
result of the run that replaced it.

`claim_next` is the difference from the ephemeral version. A plan you can only
read is not resumable: two workers would take the same task, and a worker that
dies mid-task would leave it "in progress" forever.

## What makes it survive a restart

- **Store-backed state.** Pass `Store(db="planner.sqlite")`; the default
  in-memory `Store` puts you back to a per-process plan.
- **A stable `plan_id`.** That string *is* the resume handle — same id, same
  plan.
- **Leases.** A claimed task carries an owner and a timestamp. If nobody closes
  it within `lease_seconds`, a later run may take it over: that is how work
  interrupted by a crash comes back instead of being lost.
- **Attempts.** Each claim counts. After `max_attempts` the task is parked as
  `failed` rather than handed out forever — a task that kills its worker must
  not stall the whole plan.
- **Compare-and-swap.** Every mutation is a CAS on the document, so two workers
  sharing one `Store` serialise instead of overwriting each other.

## Driving it

One wake-up should be one task. Keep each run short — that also keeps it inside
the per-run timeouts of the CLI-backed engines:

```python
with Store(db="planner.sqlite") as store:
    planner = durable_blackboard_agent([worker], store=store, plan_id="quarterly-memo")
    planner("Continue the plan: claim the next task, do it, and tick it off.")
```

Call that from a scheduler, a loop, or a LazyPulse tick. Nothing is held open
between runs, so the agent can be restarted at any moment — a scheduler that
rebuilds the whole agent on every firing needs nothing else from you, as long
as it hands the same `Store` back in.

**Size `lease_seconds` above the real task duration.** The lease exists to
recover work from a dead worker, and it cannot tell "dead" from "slow": if a
task takes longer than its lease, the next firing will reclaim it and two
workers will run the same item. Rule of thumb: longer than the slowest task,
and longer than the interval between firings.

## Using it without an agent

`DurableBlackboard` is a plain object — useful for a queue you drive yourself,
or for inspecting a plan from outside the agent:

```python
board = DurableBlackboard(store, "quarterly-memo", lease_seconds=600, max_attempts=3)
board.set_plan("quarterly review", ["gather", "extract", "write"])

claimed = board.claim_next(owner="worker-1")   # (0, "gather") or None
board.mark_done(0, "pulled 4 filings", owner="worker-1")

board.snapshot().complete                       # bool
print(board.render())                           # the same text the agent sees
```

## See also

- [Blackboard planner](blackboard-planner.md) — the ephemeral sibling.
- [Checkpoint & resume](../guides/full/checkpoint.md) — durable state for a
  *static* `Plan` DAG, when the structure is known up front.
- [Store](../guides/mid/store.md) — what backs the board.
