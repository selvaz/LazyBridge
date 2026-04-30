# Mid tier

**Use this when** you need conversation memory, shared key-value state, request/response tracing, guardrails, linear multi-agent chains, a simple human approval gate, or to wire in an MCP server as a tool catalogue.

**Move to Full when** your pipeline has conditional branching, typed hand-offs between steps, or crash-resume requirements.

## Walkthrough

Mid is where stateful, observable, multi-agent applications live.
Three concepts cover state:

* **`Memory`** — in-prompt conversation context.  What the model
  sees in the next turn.
* **`Store`** — durable cross-run state.  What survives a crash.
* **`Session`** — observability container.  Events, exporters,
  cost roll-up.

`Guards` filter input and output (cheap regex or LLM-as-judge).
Three composition primitives stack on top: `Agent.chain` for linear
pipelines, `Agent.parallel` for deterministic fan-out, and
`Agent.as_tool` (implicit when you pass an Agent as a tool) for
Agent-of-Agents trees.

The tier closes with the alpha extensions you'll likely reach for:
`MCP` to consume an external tool catalogue, `HumanEngine` for an
approval gate, `EvalSuite` for behaviour testing.  See
[core-vs-ext](../guides/core-vs-ext.md) for the stability promises.

## Topics

* [Memory](../guides/memory.md)
* [Store](../guides/store.md)
* [Session & tracing](../guides/session.md)
* [Guards](../guides/guards.md)
* [Agent.chain](../guides/chain.md)
* [Agent.as_tool](../guides/as-tool.md)
* [Agent.parallel](../guides/agent-parallel.md)
* [HumanEngine (alpha — ext.hil)](../guides/human-engine.md)
* [EvalSuite (alpha — ext.evals)](../guides/evals.md)
* [MCP integration (alpha — ext.mcp)](../guides/mcp.md)

## Also see

* [Testing (MockAgent)](../guides/testing.md) — deterministic agent doubles for unit tests
* [Core vs Extension policy](../guides/core-vs-ext.md) — what `(alpha)` means and how extensions become core

## Next steps

* Recipe: [Human-in-the-loop](../recipes/human-in-the-loop.md) · [MCP integration](../recipes/mcp.md) · [Orchestration tools](../recipes/orchestration-tools.md)
* [Full tier →](full.md) when you need typed pipelines or crash recovery

**By the end of Mid you can:**

- keep multi-turn conversation context with `Memory`
- persist cross-run state with `Store`
- emit structured events to console / JSON / OTel via `Session`
- filter input + output with `Guard*` (regex or LLM-as-judge)
- compose agents linearly (`Agent.chain`) or in fan-out (`Agent.parallel`)
- wire any MCP server in as a tool catalogue
- drop a human approval gate into any pipeline
