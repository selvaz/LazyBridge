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

Four composition patterns, picked by **who decides what runs when**:

* `Agent.chain` and `Agent.parallel` are **sugar** — deterministic,
  pre-scripted, no LLM orchestrator. Use when you know the shape.
* `Agent(tools=[a, b, c])` is **LLM-driven** — the model picks which
  tools to call and in what order; parallel execution of multiple tool
  calls in a single turn happens automatically.
* `Plan` is **declared and typed** — steps have named outputs, optional
  routing via `out.next: Literal[...]`, compile-time validation,
  checkpoint/resume via a backing Store.

The three are composable: a Plan step's target can be an Agent which
itself has `tools=[...]`, and so on down.
