## question
Checkpoint/resume: when is it worth the storage complexity?

## tree
Short-running pipeline, idempotent if rerun from scratch?
    → Don't bother — no store=, no checkpoint_key=, no resume=.

Long or expensive pipeline; partial-run survival matters?
    → Plan(..., store=Store(db="..."), checkpoint_key="...", resume=True)
      → failed step retries on resume; done pipeline short-circuits
        to cached kv.

Pipeline waits on external events (webhook, human, retry queue)?
    → Same pattern; you split the run across processes and re-enter
      on event delivery.

Dev loop iterating on a specific step?
    → Pin previous steps via checkpoint so you don't re-pay for
      them on every iteration.

Need a user-visible history of every step's Envelope?
    → Checkpoint is minimal (writes + next_step + status only).
      For full history use Session + JsonFileExporter.

## tree_mermaid
flowchart TD
    A[Is checkpoint/resume worth it?] --> B{Factors}
    B -->|short, idempotent| C[No]
    B -->|expensive, crashy, long| D[Yes — resume=True]
    B -->|async / external events| E[Yes — re-enter after event]
    B -->|dev iteration loop| F[Yes — pin upstream steps]
    B -->|need full run trace| G[Session + exporter, not checkpoint]

## notes
Enable when re-running earlier steps costs more than the storage
overhead. Checkpoint is minimal (one JSON write per step; `writes`
bucket + next step + status). It is not a full run history — for that,
combine `Plan` with `Session` + `JsonFileExporter`.
