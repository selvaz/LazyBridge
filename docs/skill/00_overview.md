# LazyBridge v1.0 — Skill overview

Tier-organised reference. Load on demand. The whole framework is one `Agent` with swappable engines; tools can be plain functions, other Agents, or Agents-of-Agents — the composition is closed and uniform. Parallelism inside an engine is automatic (the LLM or the Plan decide); no separate "parallel mode" exists.

## Pick your tier


### Basic
**Use this when** you need a single LLM call — with or without tools, with or without structured output. No setup beyond an API key.

**Move to Mid when** you need memory across calls, shared state, tracing, guardrails, or more than one agent in sequence.

Covers: Agent, Tool, Native tools (web search, code execution, …), Function → Tool (schema modes), Envelope

### Mid
**Use this when** you need conversation memory, shared key-value state, request/response tracing, guardrails, linear multi-agent chains, a simple human approval gate, or to wire in an MCP server as a tool catalogue.

**Move to Full when** your pipeline has conditional branching, typed hand-offs between steps, or crash-resume requirements.

Covers: Memory, Store, Session & tracing, Guards, Agent.chain, Agent.as_tool, Agent.parallel, HumanEngine (ext.hil), EvalSuite (ext.evals), MCP integration (ext.mcp)

### Full
**Use this when** you need a declared, multi-step pipeline: typed hand-offs, conditional routing, crash recovery via checkpoint/resume, or OTel/JSON observability.

**Stay at Mid** if your pipeline is a straight line with no typed models between steps and you don't need resume semantics.

Covers: Plan, Sentinels (from_prev / from_start / from_step / from_parallel), Parallel plan steps, SupervisorEngine (ext.hil), Checkpoint & resume, Exporters, GraphSchema, verify=

### Advanced
**Use this when** you are extending the framework itself: adding a new LLM provider, writing a custom execution engine, or serialising Plans across processes.

**Skip this tier** if you're building apps — Basic/Mid/Full cover everything user-facing.

Covers: Engine protocol, BaseProvider, Plan serialization, Provider registry, core.types

## Files

* `01_basic.md` — one-shot agents, tools, envelope
* `02_mid.md` — memory, store, session, guards, composition
* `03_full.md` — Plan, sentinels, supervisor, checkpoint
* `04_advanced.md` — engine protocol, providers, serialisation
* `05_decision_trees.md` — "when to use which"
* `06_reference.md` — flat API index
* `99_errors.md` — error → cause → fix table
