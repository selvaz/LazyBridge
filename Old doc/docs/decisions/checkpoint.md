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

Enable when re-running earlier steps costs more than the storage
overhead. Checkpoint is minimal (one JSON write per step; `writes`
bucket + next step + status). It is not a full run history — for that,
combine `Plan` with `Session` + `JsonFileExporter`.
