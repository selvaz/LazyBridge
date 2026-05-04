## question
Composing agents: chain, Agent.parallel, Plan, or tools=?

## tree
Linear pipeline, output of N becomes task of N+1?
    → Agent.chain(a, b, c)       # sugar for a linear Plan

Deterministic fan-out — run the same task on N agents, get list[Envelope]?
    → Agent.parallel(a, b, c)    # asyncio.gather sugar

Let the LLM decide which sub-agent(s) to call (possibly in parallel)?
    → Agent(tools=[a, b, c])     # the engine fans out automatically

Declared workflow with typed hand-offs, routing, resume?
    → Plan(Step(..., output=...), ...)

## tree_mermaid
flowchart TD
    A[What shape is my pipeline?] --> B{Who decides the flow?}
    B -->|it's linear and fixed| C[Agent.chain]
    B -->|I want N things at once| D[Agent.parallel]
    B -->|LLM picks which/when| E[Agent tools equals candidates]
    B -->|I declare steps with types| F[Plan + Step]
    F --> F1[Routing: Step routes or routes_by]
    F --> F2[Resume: store + checkpoint_key]
    F --> F3[Parallel step: Step parallel equals True]

## notes
Pick by **who decides what runs when**: `chain`/`parallel` are
pre-scripted; `tools=[...]` is LLM-driven; `Plan` is typed and
declared with compile-time validation. All three compose freely.
