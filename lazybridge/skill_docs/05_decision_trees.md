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

Start as low as possible. Tiers are additive — no code changes when
you move up. Advanced is for framework authors only.

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

`.text()` is always safe — serialises Pydantic payloads as JSON,
returns empty string for `None`.

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

`Memory` is per-agent conversation history with compression. `Store` is
a shared, addressable blackboard (use `db=` for persistence). `sources=`
injects any live text into the system prompt at call time. All three
compose freely on the same agent.

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

Pick by **who decides what runs when**: `chain`/`parallel` are
pre-scripted; `tools=[...]` is LLM-driven; `Plan` is typed and
declared with compile-time validation. All three compose freely.

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
           Step(join,
                task="Aggregate the branches.",
                context=[from_parallel("a"), from_parallel("b")]))
```

No serial/parallel mode switch. Automatic parallelism is always on when
the model emits multiple tool calls. Declared parallelism is when you
fix the shape yourself via `Agent.parallel` or `Step(parallel=True)`.

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

`HumanEngine` = one prompt, one answer. `SupervisorEngine` = full REPL
(continue / retry / store / tool commands). Use `input_fn=` in tests.
`verify=` is an automated LLM judge, not human-in-the-loop.

## verify= at Agent level, tool level, or Plan step level?

```
Gating the final output of a single agent?
    → Agent("model", verify=judge, max_verify=3)

One sub-agent inside a tool list is the risky one;
rest of the run is fine?
    → parent.as_tool(verify=judge, max_verify=2)   # Option B

One step of a declared Plan needs a judge; other steps don't?
    → Step(Agent(..., verify=judge), ...)

Want a judge on every tool call emitted by the model?
    → This isn't what verify= does — it's LLM-as-judge on output.
      For call-time filtering use Guards (guards.md).
```

`verify=` retries with judge feedback. Use agent-level for broad
output gates, tool-level (`as_tool(verify=...)`) when one sub-agent
is risky, Plan step-level when one step needs a gate. For filtering
every tool invocation, use a Guard instead — `verify=` gates output,
not individual calls.

## Checkpoint/resume: when is it worth the storage complexity?

```
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
```

Enable when re-running earlier steps costs more than the storage
overhead. Checkpoint is minimal (one JSON write per step; `writes`
bucket + next step + status). It is not a full run history — for that,
combine `Plan` with `Session` + `JsonFileExporter`.

## Do I actually need the Advanced tier?

```
Adding support for a new LLM vendor?
    → BaseProvider + register_provider_rule    (Yes, Advanced.)

Replacing the execution loop with a non-LLM strategy (rules, RL, etc.)?
    → Engine Protocol                          (Yes, Advanced.)

Serialising a Plan to disk / over the wire?
    → Plan.to_dict / Plan.from_dict            (Yes, Advanced.)

Importing lazybridge.core.types directly in app code?
    → Usually a smell — you probably want Envelope / Agent / Tool.
      (STOP. Revisit Full tier.)

Tweaking prompts, swapping models, building pipelines?
    → Basic / Mid / Full cover this.           (STOP. You're fine.)
```

Advanced is framework authorship, not application development. Smell
test: if you're importing from `lazybridge.core.*` in app code, step
back — `from lazybridge import ...` covers 99% of use cases.
