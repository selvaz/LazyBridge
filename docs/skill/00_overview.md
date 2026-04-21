# LazyBridge v1.0 — Skill overview

Tier-organised reference. Load on demand. The whole framework is one `Agent` with swappable engines; tools can be plain functions, other Agents, or Agents-of-Agents — the composition is closed and uniform. Parallelism inside an engine is automatic (the LLM or the Plan decide); no separate "parallel mode" exists.

## Pick your tier


### Basic
One-shot or tool-calling agents. Text or structured output.
No memory, no pipeline, no HIL. If you need state across calls
or more than one agent, go to Mid.

Covers: Agent, Tool, Envelope

### Mid
Realistic apps. Conversation memory, shared state, console/verbose
logging, input/output guardrails, simple chain or parallel fan-out,
one agent re-used as a tool, basic human-in-the-loop, evals.
No explicit DAG — for that, go to Full.

Covers: Memory, Store, Session & tracing, Guards, Agent.chain, Agent.as_tool, Agent.parallel, HumanEngine, EvalSuite

### Full
Production pipelines. Declared workflows with typed hand-offs between
steps, conditional routing via sentinels, checkpoint/resume after
crashes, OTel/JSON export, tool-level verifiers, serialisable plans.

Covers: Plan, Sentinels (from_prev / from_start / from_step / from_parallel), Parallel plan steps, SupervisorEngine, Checkpoint & resume, Exporters, GraphSchema, verify=

### Advanced
Framework extension. New providers, new execution engines,
cross-process Plan serialisation, direct use of core.types.
Skip this tier if you're not writing framework code.

Covers: Engine protocol, BaseProvider, Plan serialization, Provider registry, core.types

## Files

* `01_basic.md` — one-shot agents, tools, envelope
* `02_mid.md` — memory, store, session, guards, composition
* `03_full.md` — Plan, sentinels, supervisor, checkpoint
* `04_advanced.md` — engine protocol, providers, serialisation
* `05_decision_trees.md` — "when to use which"
* `06_reference.md` — flat API index
* `99_errors.md` — error → cause → fix table
