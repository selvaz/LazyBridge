# Checkpoint/resume: when is it worth the storage complexity?

```mermaid
flowchart TD
    A[Is checkpoint/resume worth it?] --> B{Factors}
    B -->|short, idempotent| C[No]
    B -->|expensive, crashy, long| D[Yes — resume=True]
    B -->|async / external events| E[Yes — re-enter after event]
    B -->|dev iteration loop| F[Yes — pin upstream steps]
    B -->|need full run trace| G[Session + exporter, not checkpoint]
```

Rule of thumb: enable checkpointing when the cost of re-running earlier
steps exceeds the cost of the storage complication.

Cost of checkpointing is low: one JSON write per step, persistence via
SQLite WAL, minimal state shape (`writes` bucket + next step + status).
It is **not** a full run history — the in-memory `StepResult` history
is rebuilt empty on resume. If you need the full audit trail, combine
`Plan` with a `Session` + `JsonFileExporter`.
