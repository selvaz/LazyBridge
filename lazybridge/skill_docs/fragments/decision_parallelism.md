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
           Step(join, task=from_parallel("a")))   # plus aggregation step

## tree_mermaid
flowchart TD
    A[Who decides the parallelism shape?] --> B{LLM or you?}
    B -->|the LLM, emergent| C[Agent tools equals candidates]
    B -->|me, deterministic fan-out| D[Agent.parallel]
    B -->|me, part of a typed workflow| E[Plan + Step parallel equals True]

## notes
Parallelism is not a configuration knob in LazyBridge. It happens in
one of two ways:

1. **Automatic (LLM-driven).** When an engine's underlying model emits
   multiple tool calls in a single turn, they execute concurrently via
   `asyncio.gather`. You do not opt in — you just pass the candidates
   in `tools=[...]`. This covers "call `search` and `calc` in the same
   step" scenarios.

2. **Declared.** You wrote the shape. Either as `Agent.parallel(a, b, c)`
   (pre-scripted fan-out returning `list[Envelope]`) or as a `Plan` with
   `Step(parallel=True)` (declared concurrent branches in a typed DAG,
   joined by `from_parallel`).

There is **no "serial vs parallel mode"** on `LLMEngine`. The old
`tool_choice="parallel"` option is deprecated — LazyBridge always
dispatches concurrently.
