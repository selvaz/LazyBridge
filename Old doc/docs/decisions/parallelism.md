# Parallelism: automatic or declared?

```mermaid
flowchart TD
    A[Who decides the parallelism shape?] --> B{LLM or you?}
    B -->|the LLM, emergent| C[Agent tools equals candidates]
    B -->|me, deterministic fan-out| D[Agent.parallel]
    B -->|me, part of a typed workflow| E[Plan + Step parallel equals True]
```

No serial/parallel mode switch. Automatic parallelism is always on when
the model emits multiple tool calls. Declared parallelism is when you
fix the shape yourself via `Agent.parallel` or `Step(parallel=True)`.
