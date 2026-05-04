# Getting started

LazyBridge has one public abstraction — `Agent` — that you compose into
progressively richer pipelines. This page walks the four levels of
sophistication so you can pick the smallest one that fits your problem
and skim the rest.

| Level     | Add this when…                                                    | Move on when…                                              |
|-----------|-------------------------------------------------------------------|------------------------------------------------------------|
| **Basic** | one LLM call, with or without tools / structured output           | you need state across calls, multiple agents, or tracing   |
| **Mid**   | conversation memory, shared store, tracing, guardrails, MCP       | your pipeline needs typed steps, branching, or crash-resume |
| **Full**  | declared multi-step `Plan`, sentinels, checkpoint/resume, OTel    | you need a custom provider or execution engine             |
| **Advanced** | implementing a new provider, engine, or persisting Plans across processes | you're extending the framework itself                |

Each level builds on the previous one; nothing earlier becomes invalid.

---

## Basic — one call

**Use this when** you need a single LLM call — with or without tools,
with or without structured output. No setup beyond an API key.

An `Agent` is a thin wrapper that delegates to an `Engine`. The default
engine is `LLMEngine`, so `Agent("claude-opus-4-7")` is everything you
need for a one-shot call.

**Tools** are the next concept. Pass any Python function with type
hints and a docstring as `tools=[...]` and the framework builds the
JSON schema automatically — see [Function → Tool](tool-schema.md) for
when the signature path isn't enough. **Native tools** are
provider-hosted alternatives (web search, code execution): pass an
enum, no code.

Every call returns an **`Envelope`** — a typed wrapper carrying
`payload` (the result), `metadata` (token / cost / latency), and
`error`. Use `.text()` for "give me a string"; `.payload` for the
typed Pydantic model when `output=` is set; `.ok` to check error state.

### Topics
* [Agent](agent.md)
* [Tool](tool.md)
* [Native tools (web search, code execution, …)](native-tools.md)
* [Function → Tool (schema modes)](tool-schema.md)
* [Envelope](envelope.md)

### Recipes
* [Tool calling](../recipes/tool-calling.md) · [Structured output](../recipes/structured-output.md)

**By the end of Basic you can:**

- call any LLM with `Agent("model")(task)`
- turn any typed Python function into a tool with auto-schema
- opt into provider-hosted native tools (web search, code execution)
- get a typed Pydantic payload back instead of plain text

---

## Mid — state, observability, multi-agent

**Use this when** you need conversation memory, shared key-value
state, request/response tracing, guardrails, linear multi-agent
chains, a simple human approval gate, or to wire in an MCP server as
a tool catalogue.

Three concepts cover state:

- **`Memory`** — in-prompt conversation context. What the model sees
  in the next turn.
- **`Store`** — durable cross-run state. What survives a crash.
- **`Session`** — observability container. Events, exporters, cost
  roll-up.

`Guards` filter input and output (cheap regex or LLM-as-judge). Three
composition primitives stack on top: `Agent.chain` for linear
pipelines, `Agent.parallel` for deterministic fan-out, and
`Agent.as_tool` (implicit when you pass an Agent as a tool) for
Agent-of-Agents trees.

The tier closes with the framework extensions you'll likely reach
for: `MCP` to consume an external tool catalogue, `HumanEngine` for
an approval gate, `EvalSuite` for behaviour testing. See
[core-vs-ext](core-vs-ext.md) for the namespace layout and import
boundaries.

### Topics
* [Memory](memory.md)
* [Store](store.md)
* [Session & tracing](session.md)
* [Guards](guards.md)
* [Agent.chain](chain.md)
* [Agent.as_tool](as-tool.md)
* [Agent.parallel](agent-parallel.md)
* [HumanEngine (ext.hil)](human-engine.md)
* [EvalSuite (ext.evals)](evals.md)
* [MCP integration (ext.mcp)](mcp.md)

### Also see
* [Testing (MockAgent)](testing.md) — deterministic agent doubles for unit tests
* [Core vs Extension policy](core-vs-ext.md) — pre-1.0 alpha posture and the namespace layout

### Recipes
* [Human-in-the-loop](../recipes/human-in-the-loop.md) · [MCP integration](../recipes/mcp.md) · [Orchestration tools](../recipes/orchestration-tools.md)

**By the end of Mid you can:**

- keep multi-turn conversation context with `Memory`
- persist cross-run state with `Store`
- emit structured events to console / JSON / OTel via `Session`
- filter input + output with `Guard*` (regex or LLM-as-judge)
- compose agents linearly (`Agent.chain`) or in fan-out (`Agent.parallel`)
- wire any MCP server in as a tool catalogue
- drop a human approval gate into any pipeline

---

## Full — declared pipelines, crash-resume

**Use this when** you need a declared, multi-step pipeline: typed
hand-offs, conditional routing, crash recovery via checkpoint/resume,
or OTel/JSON observability.

A `Plan` is a list of `Step`s validated at construction time —
`PlanCompileError` fires before any LLM call on a misspelled step
name, a forward `from_step`, or a parallel-band misuse.

**Sentinels** (`from_prev`, `from_start`, `from_step("name")`,
`from_parallel`, `from_parallel_all`) are how each step declares
where its input and context come from — `task=` for the prompt
instruction, `context=` for the data (which can be a list of
sentinels for multi-source synthesis without a combiner step).

`parallel=True` marks a step as a member of a concurrent band.
`writes="key"` persists the step's payload to the `Store` so a
downstream agent (or a future resume) can read it. Pair `Plan` with
`Store` + `checkpoint_key=` + `resume=True` to get CAS-protected
crash-resume; pair with `on_concurrent="fork"` for fan-out workflows.

`SupervisorEngine` is the HIL counterpart: a step that hands control
to a human REPL. `verify=` adds a judge/retry loop at either
agent-level or tool-level. `Exporters` and `GraphSchema` surface the
run as OTel spans + a topology graph for dashboards.

Before shipping, read the [Operations checklist](operations.md):
back-pressure, OTel GenAI conventions, `Memory.summarizer_timeout`,
MCP cache TTL, and the production knobs on `Agent` (`timeout`,
`cache`, `fallback`).

### Topics
* [Plan](plan.md)
* [Sentinels (from_prev / from_start / from_step / from_parallel)](sentinels.md)
* [Parallel plan steps](parallel-steps.md)
* [SupervisorEngine (ext.hil)](supervisor.md)
* [Checkpoint & resume](checkpoint.md)
* [Exporters](exporters.md)
* [GraphSchema](graph-schema.md)
* [verify=](verify.md)

### Also see
* [Operations checklist](operations.md) — production deployment knobs

### Recipes
* [Plan with typed steps and crash resume](../recipes/plan-with-resume.md)

**By the end of Full you can:**

- declare a typed multi-step pipeline that fails-fast at construction (`PlanCompileError`)
- route conditionally via `Step(routes={...})` predicates or `Step(routes_by="field")`
- run parallel bands and aggregate them via `from_parallel_all` or `context=[...]`
- resume from the failed step with CAS-protected checkpoints
- run N copies of the same pipeline concurrently with `on_concurrent="fork"`
- swap in a `SupervisorEngine` to give a human a tool-calling REPL mid-pipeline
- add a `verify=` judge at agent-level or tool-level

---

## Advanced — extending the framework

**Use this when** you are extending the framework itself: adding a
new LLM provider, writing a custom execution engine, or serialising
Plans across processes.

The `Engine` protocol is one method (`run`) and one optional one
(`stream`); implement it to add a new execution model alongside
`LLMEngine` / `Plan` / `HumanEngine` / `SupervisorEngine`.

`BaseProvider` is the contract every LLM provider satisfies — four
methods, full retry/backoff handled for you by `Executor`. The
`Provider registry` is the runtime entry point: register aliases /
rules so `Agent("my-model")` routes to the new provider without
forking the framework. The `LiteLLM bridge` is the catch-all
alternative — unlock 100+ extra providers via the `litellm/<model>`
prefix without writing a provider class.

`Plan.to_dict` / `from_dict` round-trip a Plan's topology through
JSON for cross-process re-use; callables and Agents are rebound via
`registry={name: target}`. `core.types` carries the types that flow
across all these boundaries.

### Topics
* [Engine protocol](engine-protocol.md)
* [BaseProvider](base-provider.md)
* [Plan serialization](plan-serialize.md)
* [Provider registry](register-provider.md)
* [core.types](core-types.md)

### Also see
* [LiteLLM bridge](litellm.md) — 100+ extra providers via `litellm/<model>`
* [Core vs Extension policy](core-vs-ext.md) — pre-1.0 alpha posture and the import boundary

**By the end of Advanced you can:**

- implement a brand-new LLM provider against `BaseProvider`
- implement a brand-new execution engine against the `Engine` protocol
- serialise a `Plan` to JSON for cross-process re-use
- register custom model-string aliases at runtime
