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

## The verbs

| Tool | What it does |
|---|---|
| `set_plan(reasoning, tasks)` | Creates the plan. **Refuses** to discard one still in progress. |
| `get_plan()` | The whole board: what is done, claimed, failed, and what is next. |
| `add_tasks(tasks)` | Appends newly-discovered tasks without touching any existing task's index or status. |
| `cancel_task(index, expected_text, reason)` | Drops a not-yet-claimed task that turned out to be unnecessary. `expected_text` must match the task's current text. |
| `set_task_schedule(index, expected_text, planned_start_at, due_at, reason)` | Sets (or clears, by passing `None`) a `todo`/`claimed` task's planned start and due timestamps. Same `expected_text` stale-index guard; each call also appends an entry to an audit trail. |
| `claim_next()` | Takes exactly one task — whichever is earliest eligible — atomically. |
| `claim_task(index, expected_text, renew=False)` | Takes a SPECIFIC task instead of "next", for a caller that already knows which one it needs and doesn't want to claim-and-close every earlier task just to reach it. Same `expected_text` stale-index guard as `cancel_task`. |
| `mark_done(index, summary)` | Closes a task with its result. |
| `mark_failed(index, error)` | Hands a task back after a genuine failure. |

### Re-entering a claim you already hold (`renew=True`)

By default `claim_task` **refuses** a task that is currently claimed, even by
the caller that claimed it: two callers are only ever "the same" if one of
them says so. Passing `renew=True` lets the *same identified owner* re-enter
a claim it already holds -- the shape of a flow that claims a task, does some
preparation, and claims the same index again to attach the job that will do
the work.

Renewal is deliberately narrow. It succeeds only when **all** of these hold,
and is refused otherwise:

- `owner=` was given and equals the current holder (`owner=None` never
  renews: two anonymous callers are not the same caller);
- `renew=True` was passed. An owner is usually *process*-level, so equality
  alone cannot tell "the holder re-entering" from "a second, genuinely new
  delegation for a task that already has one running" -- inferring renewal
  would let one batch start two writers on one task;
- the lease is still **alive**. A holder whose lease lapsed did not finish in
  time, and that is exactly what an attempt counts; renewing it would let a
  stable owner identity dodge the attempt limit forever;
- the claim has not **already been renewed**. A renewal attaches a worker to
  a task; a second one would attach a second worker to the same checkout.
  The check happens inside the compare-and-swap, so two concurrent callers
  cannot both win.

A fresh claim always starts un-renewed, whichever way it was taken
(`claim_next` included).

Closing requires an **active claim**: a task that was never claimed, or one
already closed, is refused. Each planner instance also claims under its own
identity, so a run whose lease expired cannot come back and overwrite the
result of the run that replaced it.

`claim_next` is the difference from the ephemeral version. A plan you can only
read is not resumable: two workers would take the same task, and a worker that
dies mid-task would leave it "in progress" forever.

## Scheduling, `completed_at`, and OVERDUE

Every task carries `planned_start_at` and `due_at` (both `None` until set) and
a `completed_at` timestamp, stored on the task record but **not** shown by
`render()` — a consumer that reads the plan document directly (not just
`get_plan()`'s text) uses it to place a closed task in time. Only
`set_task_schedule` writes the first two — `set_plan`/`add_tasks` always
create a task with both `None` — and only a task actually reaching a
**terminal** state (`done`, an exhausted `failed`, or `cancelled`) gets
`completed_at` set; a `mark_failed` that sends a task back to `todo` for a
retry is not a close and leaves it `None`.

`set_task_schedule` requires `reason`, rejects a non-finite or `bool` value
for either timestamp, and rejects `planned_start_at > due_at` when both are
given. It only accepts `todo`/`claimed` tasks — a terminal task can't be
(re)scheduled. Every call, including one that only clears a timestamp with
`None`, appends one entry to the plan's own append-only `schedule_events`
list — `{task_index, planned_start_at, due_at, reason, at}` — written in the
**same** compare-and-swap as the task update itself, so a concurrent close of
that task either wins the race first (and this edit is rejected on retry
against the now-terminal task) or the schedule and its audit entry land
together.

`render()` (what `get_plan()`/the CLI show) prints a task's planned interval
under its line when set, and appends `OVERDUE` next to `due_at` when it's in
the past **and** the task is still `todo` or `claimed` — never for a task
that already closed, on time or not.

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
