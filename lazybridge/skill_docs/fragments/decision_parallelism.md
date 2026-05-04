## question
Parallelism: automatic or declared?

## tree
Do you want the LLM to decide whether to run things in parallel?
    → Pass them in tools=[...] on a plain Agent. When the model emits
      multiple tool calls in one turn, LazyBridge runs them concurrently
      via asyncio.gather. No configuration.

Do you want to declare that N agents run at once on the same task?
    → Agent.parallel(a, b, c)(task)   # → list[Envelope]

Do you want declared concurrent branches inside a typed workflow?
    → Plan(Step(a, parallel=True),
           Step(b, parallel=True),
           Step(join,
                task="Aggregate the branches.",
                context=[from_parallel("a"), from_parallel("b")]))

## tree_mermaid
flowchart TD
    A[Who decides the parallelism shape?] --> B{LLM or you?}
    B -->|the LLM, emergent| C[Agent tools equals candidates]
    B -->|me, deterministic fan-out| D[Agent.parallel]
    B -->|me, part of a typed workflow| E[Plan + Step parallel equals True]

## notes
No serial/parallel mode switch. Automatic parallelism is always on when
the model emits multiple tool calls. Declared parallelism is when you
fix the shape yourself via `Agent.parallel` or `Step(parallel=True)`.
