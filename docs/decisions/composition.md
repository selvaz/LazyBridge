# Composing agents: chain, Agent.parallel, Plan, or tools=?

```mermaid
flowchart TD
    A[What shape is my pipeline?] --> B{Who decides the flow?}
    B -->|it's linear and fixed| C[Agent.chain]
    B -->|I want N things at once| D[Agent.parallel]
    B -->|LLM picks which/when| E[Agent tools equals candidates]
    B -->|I declare steps with types| F[Plan + Step]
    F --> F1[Routing: out.next literal]
    F --> F2[Resume: store + checkpoint_key]
    F --> F3[Parallel step: Step parallel equals True]
```

Pick by **who decides what runs when**: `chain`/`parallel` are
pre-scripted; `tools=[...]` is LLM-driven; `Plan` is typed and
declared with compile-time validation. All three compose freely.
