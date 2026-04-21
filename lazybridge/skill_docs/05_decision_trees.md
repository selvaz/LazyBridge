# LazyBridge — Decision trees

When to use which part of the framework. Each tree answers a concrete question you hit while building.

## Where do I start: Basic, Mid, Full, or Advanced?

```
Single agent, one call, maybe with tools?
    → Basic.   Agent("model")(task).text()   or   tools=[fn]

Need conversation memory / shared state / tracing / guardrails
or simple chain/parallel composition?
    → Mid.     Memory, Store, Session(console=True), guards,
               Agent.chain, Agent.parallel, agent.as_tool(), HumanEngine

Declared multi-step workflow with typed hand-offs / routing /
resume after crashes / OTel export / tool-level verifiers?
    → Full.    Plan, Step, sentinels, SupervisorEngine,
               checkpoint/resume, exporters, verify=

Writing framework code — new provider, new engine,
cross-process Plan serialisation, direct core.types use?
    → Advanced.
```

Start as low as possible. Moving up a tier is additive — no code you
wrote in Basic needs to change when you later add Memory (Mid) or wrap
the agent in a Plan (Full). Most users live in Basic and Mid; Full is
for production-grade declared pipelines; Advanced only applies if you're
writing framework code.

## What does my agent return — text, typed object, or metadata?

```
Plain string response?
    → .text()                 # str, always, regardless of output type

Pydantic model / structured result?
    → Agent("model", output=MyModel)(task).payload   # MyModel instance

Need token count, cost, latency, run id?
    → env.metadata            # EnvelopeMetadata dataclass

Need to check for errors before reading payload?
    → if env.ok: ... else: env.error.message
```

An ``Envelope`` carries everything the engine knows about a run: the
payload (string by default; typed when ``output=`` is set), metadata
(tokens, cost, latency, run id), and an optional error channel. You
pick what you read; nothing is hidden.

Calling ``.text()`` is safe on every Envelope — it serialises Pydantic
payloads as JSON and handles ``None`` as empty string.

## State: Memory, Store, or sources=?

```
Conversation history for one agent?
    → Memory         # agent.memory = Memory("auto")

Shared key-value blackboard across multiple agents / runs?
    → Store          # store.write / store.read

Static documents (files, URLs, strings) injected into context at call time?
    → sources=[...]  # callable, Memory, Store, or raw string

Multiple patterns at once?
    → Yes — they compose.
      Example: Agent(memory=Memory(), sources=[shared_store, policy_text])
```

`Memory` is conversational and per-agent. It records turns and
compresses older ones when your token budget is exceeded.

`Store` is a blackboard: explicit, addressable by key, shareable.
Use it when agents need to hand off intermediate state or cache results
across runs. Pass ``db="file.sqlite"`` for persistence.

`sources=[...]` is context injection. Each source object is asked for
its current text at call time (live view — no snapshotting), and the
concatenated text is appended to the system prompt. Sources can be
`Memory`, `Store`, callables, or plain strings.

All three compose: an agent can have its own `memory`, read from a
shared `Store` via `sources=`, and also inject a policy string.

## Composing agents: chain, Agent.parallel, Plan, or tools=?

```
Linear pipeline, output of N becomes task of N+1?
    → Agent.chain(a, b, c)       # sugar for a linear Plan

Deterministic fan-out — run the same task on N agents, get list[Envelope]?
    → Agent.parallel(a, b, c)    # asyncio.gather sugar

Let the LLM decide which sub-agent(s) to call (possibly in parallel)?
    → Agent(tools=[a, b, c])     # the engine fans out automatically

Declared workflow with typed hand-offs, routing, resume?
    → Plan(Step(..., output=...), ...)
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

## Parallelism: automatic or declared?

```
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
```

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

## Human-in-the-loop: HumanEngine or SupervisorEngine?

```
Simple "wait for input / approve / fill a form"?
    → HumanEngine                 # terminal prompt, optional Pydantic form

Interactive REPL where the human can call tools, retry agents
with feedback, and inspect the store?
    → SupervisorEngine            # commands: continue, retry <agent>: <fb>,
                                  #           store <key>, <tool>(<args>)

Automated (no human) verification at runtime?
    → verify=judge_agent          # NOT a HIL paradigm — it's an LLM judge
```

`HumanEngine` is the minimum: one prompt, one answer. Good for
approvals, reviewers, light annotation, Pydantic forms. Blocks on
`input()` or a custom UI adapter.

`SupervisorEngine` is a full REPL. The human sees the previous output
and can:

* `continue` — accept and return to the pipeline,
* `retry <agent>: <feedback>` — re-run a registered agent with feedback,
* `store <key>` — inspect the shared Store,
* `<tool>(<args>)` — invoke a registered tool directly.

Use it when the human is an operator in the pipeline, not just a
gate. Pass `input_fn=_scripted(...)` in tests to keep the loop
non-interactive.

`verify=` is **not** HIL at all — the judge is an Agent, automated.
Listed here because users often conflate "verification" with "human
review".
