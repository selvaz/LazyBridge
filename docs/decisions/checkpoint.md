# Checkpoint & resume

> **When is the storage overhead worth it?**

Checkpointing writes one JSON snapshot per step. Worth it when
re-running an early step costs more than the storage round-trip;
not worth it for short, idempotent pipelines.

## Decision tree

```text
Short-running pipeline, idempotent if rerun from scratch?
    → No checkpoint. Plan(*steps)  — without store= / checkpoint_key=.

Long or expensive pipeline, partial-run survival matters?
    → Plan(*steps,
           store=Store(db="run.sqlite"),
           checkpoint_key="run-2026-04-30",
           resume=True)
      # Failed step retries on resume; "done" pipeline short-
      # circuits to the cached writes-bucket.

Pipeline waits on external events (webhook, human approval, retry queue)?
    → Same pattern; you split the run across processes and re-enter
      with resume=True after the event is delivered.

Dev loop iterating on a specific step?
    → Pin upstream steps via the same checkpoint_key so you don't
      re-pay for them on every iteration.

Need a user-visible history of every step's Envelope?
    → Checkpoint is minimal — only writes-bucket + next_step + status.
      For full history use Session(exporters=[JsonFileExporter("…")])
      and query session.events.query(...).
```

## Quick reference

| Situation | Use checkpoint? |
|---|---|
| Short, idempotent pipeline | **No** |
| Expensive, crash-prone, long-running | **Yes** — `store=` + `checkpoint_key=` + `resume=True` |
| Async / event-driven (re-enters across processes) | **Yes** — same pattern |
| Dev iteration loop on a specific step | **Yes** — pin upstream via checkpoint |
| Want a full run trace for audit | **No** — use `Session` + `JsonFileExporter` |
| Concurrent fan-out runs sharing one Plan shape | `on_concurrent="fork"` (no resume) |

## Notes

- **Checkpoint is minimal.** One JSON write per step: the
  `writes`-bucket payload, the next step name, the status
  (`claimed` / `running` / `failed` / `done`), the run UID, and
  (v2 only) the serialised step-result history. Not a full audit
  trail — for that pair `Plan` with `Session`.
- **Concurrent runs sharing a key are serialised.** The default
  `on_concurrent="fail"` raises `ConcurrentPlanRunError` on
  collision. Use `on_concurrent="fork"` to give each run its own
  keyspace (incompatible with `resume=True`).
- **Checkpoint writes happen *before* durable Store writes.**
  Eliminates double-writes on resume; the inverse trade-off is
  that a crash in the gap loses the durable Store value. The
  value still lives in the checkpoint's `kv` so the Plan
  continues correctly — but sidecar consumers reading the Store
  directly should reconcile against the checkpoint snapshot
  (mechanically: `Plan.store_write_is_current(store,
  checkpoint_key=..., key=...)`).
- **A failed parallel band points the checkpoint at the band's
  *first* step.** The whole band re-runs cleanly so all sibling
  `writes` are produced consistently. Branches with non-
  idempotent side effects need idempotency keys.

## See also

- [Checkpoint & resume](../guides/full/checkpoint.md) — full
  reference: persisted shape, state transitions, sidecar
  consumer rules.
- [Store](../guides/mid/store.md) — the durable layer behind
  checkpoints; SQLite WAL mode for thread-safe concurrent
  access.
- [Parallel plan steps](../guides/full/parallel-plan-steps.md)
  — band atomicity rules that drive the "next_step points to
  the band's first step" behaviour on failure.
