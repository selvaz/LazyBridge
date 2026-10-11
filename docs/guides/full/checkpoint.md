# Checkpoint & resume

Durable state for long-running plans. After every step (success or
fail) `Plan` writes its execution state to a `Store`; a re-run with
`resume=True` picks up at the failed step rather than re-running
the whole pipeline. Concurrent runs sharing a `checkpoint_key` are
serialised by compare-and-swap.

## Signature

```python
from lazybridge import Agent, Plan, Step, Store

Plan(
    *steps,
    store,                         # required for checkpointing
    checkpoint_key,                # required — unique key per run identity
    resume=False,                  # True → pick up at the failed step
    on_concurrent="fail",          # "fail" | "fork"
    max_iterations=100,            # safety valve — caps routing loops;
                                   # the resumed run continues the counter
                                   # from the checkpoint, so a resumed
                                   # router can't silently exceed it.
)


# Persisted shape at store[checkpoint_key] (CHECKPOINT_VERSION = 2).
{
    "next_step": "step_name" | None,
    "kv": {"writes_key": payload, ...},
    "completed_steps": [...],
    "status": "claimed" | "running" | "paused" | "failed" | "done",
    "run_uid": "<hex>",            # CAS ownership stamp; identifies the writer
    "checkpoint_version": 2,
    "history": [...],              # serialised StepResult list (v2 only)
}


# Errors
ConcurrentPlanRunError             # CAS collision when on_concurrent="fail"
PlanCompileError                   # on_concurrent="fork" + resume=True (incompatible)
```

## Synopsis

`Plan` writes its state to `store[checkpoint_key]` after every step.
The persisted object captures three things:

- `next_step` — the name of the step the plan would run next. On
  success, this advances; on failure, it stays pointing at the
  failing step (or, for parallel bands, at the **band's first**
  step).
- `kv` — every step's `writes="key"` payload. This is what survives
  across runs; in-memory step history is rebuilt empty on resume.
- `completed_steps` + `status` — bookkeeping for the resume logic
  to decide what to skip.

Five state transitions:

- **Claimed** — `status="claimed"`, transient. Written via CAS
  before any step runs so two concurrent fresh runs collide here
  rather than corrupting each other later. You'll only see this
  status if you inspect the store mid-run.
- **Running** — `status="running"`, `next_step=<the next step>`.
  Normal progression after a successful step.
- **Failed** — `status="failed"`, `next_step=<failing step>` (or
  the parallel band's first step). A subsequent `resume=True` run
  retries from there.
- **Paused** — `status="paused"`, `next_step=<paused step>` (or
  the parallel band's first step). Written when a step raises
  `PlanPaused` to signal a cooperative halt. A subsequent
  `resume=True` run re-invokes the paused step. Use this when a
  step has detected that an external precondition isn't met
  (webhook hasn't arrived, human approval pending) and the
  pipeline cannot proceed yet — distinct from `failed` which
  signals an actual error.
- **Done** — `status="done"`, `next_step=None`. A subsequent
  `resume=True` short-circuits and returns an envelope whose
  payload is the cached `kv`.

The persisted `history` field (v2 only) is a serialised
`StepResult` list — a resumed run rebuilds in-memory step history
from it, which is what makes `from_parallel_all` and the nested-
cost rollup behave correctly across crash boundaries. v1
checkpoints (no `history` key) degrade gracefully: in-memory
history starts empty and the parallel-band aggregator falls back
to the start envelope (legacy pre-W1.3 behaviour, no crash).

`on_concurrent` controls what happens when two runs try to use the
same `checkpoint_key` at once:

- `"fail"` (default) — single-writer semantics. The second run
  raises `ConcurrentPlanRunError`. Pair with `resume=True` for
  crash recovery.
- `"fork"` — each run claims an isolated keyspace
  `f"{checkpoint_key}:{run_uid}"`. **Incompatible with
  `resume=True`** (raises `PlanCompileError` at construction). Use
  for fan-out workflows where many runs share the same plan
  shape.

## When to use it

- **Long-running pipelines.** Anything where re-running every step
  on failure costs real time or money. Crash-resume turns a
  one-step crash into a one-step retry.
- **Pipelines with expensive early steps and cheap late steps.**
  The classic shape: `extract` (slow ETL) → `transform` (fast) →
  `load` (fast). A failure in `load` shouldn't re-run `extract`.
- **Fan-out workflows on a shared store.**
  `on_concurrent="fork"` lets you run many variants of the same
  plan against the same Store without collisions.
- **Debugging in production.** A failed run leaves the partial
  state on disk. You can inspect `store.read("…")` keys before
  the resume run to understand what the pipeline saw at the
  failure point.

## When NOT to use it

- **Short, cheap pipelines.** A 3-step pipeline of LLM calls under
  a few seconds doesn't benefit from the persistence overhead.
- **Using checkpoint persistence as side-effect rollback.** Checkpoints retain
  step history and writes, including plans without writes declarations. They
  do not undo effects in external systems.
- **Side effects that aren't crash-safe to repeat.** A failing
  step is retried as-is on `resume=True`; an external HTTP POST
  with no idempotency key may run twice. Gate with idempotency
  keys, deduplication, or a marker write before the side effect.

## Example

```python
from lazybridge import Agent, Envelope, Plan, Step, Store
from lazybridge.testing import MockAgent

extract = MockAgent(["raw records"], name="extract")
transform = MockAgent(["clean records"], name="transform")
validate = MockAgent([
    Envelope.error_envelope(RuntimeError("temporary validation failure")),
    "validated records",
], name="validate")
load = MockAgent(["loaded records"], name="load")
store = Store(db="pipeline.sqlite")

def build_plan() -> Agent:
    return Agent(
        name="etl",
        engine=Plan(
            Step(extract, writes="raw"),
            Step(transform, writes="clean"),
            Step(validate, writes="verdict"),
            Step(load, writes="loaded"),
            store=store, checkpoint_key="etl-demo", resume=True,
        ),
    )

first = build_plan()("today's batch")
assert not first.ok
state = store.read("etl-demo")
assert state["status"] == "failed"
assert state["next_step"] == "validate"
assert state["completed_steps"] == ["extract", "transform"]
assert state["kv"]["clean"] == "clean records"

second = build_plan()("today's batch")
assert second.ok
assert extract.call_count == transform.call_count == 1
assert validate.call_count == 2
assert load.call_count == 1

result = build_plan()("today's batch")
assert result.ok
assert validate.call_count == 2  # Cached done state; no repeated model/tool call.
print(result.payload)
store.close()
```

### Isolated concurrent runs

For independent batches that share a Store, fork gives every run its own
suffixed checkpoint key. Workers here are deterministic mock agents; no model
or external side effect is involved:

```python
from concurrent.futures import ThreadPoolExecutor
from lazybridge import Agent, Plan, Step, Store
from lazybridge.testing import MockAgent

store = Store(db="fanout.sqlite")
load = MockAgent("prices loaded", name="load")
run = MockAgent("trade log computed", name="run")
score = MockAgent("score computed", name="score")
backtest = Agent(name="backtest", engine=Plan(
    Step(load, writes="prices"), Step(run, writes="trades"), Step(score),
    store=store, checkpoint_key="backtest", on_concurrent="fork",
))
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(backtest, ["AAPL", "GOOG", "MSFT", "AMZN"]))
assert len(results) == 4 and all(item.ok for item in results)
store.close()
```

Fork isolation prevents checkpoint-key collisions; it does not isolate external
systems called by workers. For restart resume, choose a stable unique key per
batch with on_concurrent="fail" and resume=True. Fork and resume=True are
incompatible because there is no single shared key to resume.

### Inspecting partial state

The failure/resume example above reads status, next_step, completed_steps and
kv immediately after failure. Use those records to identify the failed boundary
and inspect stored writes before retrying. The second run skips completed work;
the third returns the cached done state. A checkpoint is not an external action
rollback or a guarantee that an interrupted worker never performed a side effect.

## Pitfalls

- **Changing the Plan and resuming from an old checkpoint will
  fail.** The saved `next_step` may no longer exist. After
  refactoring steps, delete the checkpoint:
  `store.delete(checkpoint_key)`.
- **Non-JSON-serialisable `writes=` values are stringified.** The
  Store JSON-encodes via `json.dumps(default=str)`; a file handle
  becomes its `repr(...)`. Prefer primitives, dicts, and Pydantic
  models (`.model_dump()`-friendly).
- **Resume does not re-inject `session=` or exporters.** Pass the
  same `session=` + `store=` on every run — the Plan only
  persists what's behind the `Store` interface, not the live
  observability wiring.
- **A failed parallel band points the checkpoint at the band's
  *first* step.** The whole band re-runs cleanly so all sibling
  `writes` are produced consistently — resuming mid-band would
  leave earlier branches' Store keys stale relative to the re-run
  ones.
- **`on_concurrent="fork"` + `resume=True` is a configuration
  error.** Fork mode gives each run its own key, so there's no
  shared checkpoint to resume from. The framework raises
  `PlanCompileError` at construction.
- **`ConcurrentPlanRunError` is a runtime error, not a compile
  error.** Two processes opening the same SQLite file with the
  same `checkpoint_key` and `on_concurrent="fail"` collide via
  CAS; the loser raises. Catch it explicitly if you want graceful
  retry-after-backoff semantics.
- **Cached "done" runs still cost the storage round-trip.** A
  short-circuited run returns instantly but still hits the Store
  to read the cached kv. For very high read rates, layer your
  own in-process cache.
- **Sidecar consumers should reconcile against the checkpoint
  snapshot, not the raw Store.** Each step writes its checkpoint
  *before* the durable `store.write(step.writes, value)` call —
  this eliminates double-writes on resume (the checkpoint
  already records `next_step` past the completed step). The
  trade-off is that a crash in the gap between the two writes
  loses the durable Store value; the value still lives in the
  checkpoint's serialised `kv` and is read back on resume, so
  the Plan continues correctly. Anything reading the Store
  out-of-band (a dashboard, a sidecar process) can now check
  mechanically: every plan Store write is stamped with the owning
  run's id, and `Plan.store_write_is_current(store,
  checkpoint_key=..., key=...)` returns `False` whenever the value
  is stale relative to the checkpoint — fall back to the
  checkpoint's `kv` snapshot in that case.

## See also

- [Plan](plan.md) — the engine that writes checkpoints; covers
  `max_iterations`, `on_concurrent`, and the full DAG validation
  surface.
- [Store](../mid/store.md) — the durable layer behind
  checkpoints; the SQLite WAL mode is what makes concurrent
  reads / writes safe.
- [Step](step.md) — `writes="key"` is what survives across runs;
  no `writes=` means no resume value.
- [Parallel plan steps](parallel-plan-steps.md) — the band-level
  atomicity rule that drives the "next_step points to the band's
  first step on failure" behaviour.
