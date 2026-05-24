# LazyBridge

A zero-boilerplate, multi-provider Python framework for composing LLMs, plain
functions, deterministic plans, humans, and external tools under one uniform
model: **everything is a tool.**

```python
from lazybridge import Agent

# Tier alias works across all 4 providers: anthropic / openai / google / deepseek.
agent = Agent.from_provider("anthropic", tier="cheap")
result = agent("What's the capital of France?")
print(result.text())
```

That's the whole framework's surface area when you start. It grows only when
your problem grows.

## What LazyBridge gives you

An agent in LazyBridge is the composition of three things — and only these three:

- **Engine** — the decision-making layer (an LLM, a deterministic `Plan`, a
  human-in-the-loop, or your own).
- **Tools** — every capability the agent can use. Plain Python functions,
  other agents, MCP servers, and full pipelines all behave the same way.
- **State** — `Memory`, `Session`, and `Store`: continuity, traceability, and
  shared blackboard between steps.

That's it. Whether you're writing a one-shot helper or a checkpointed
multi-region pipeline with human approvals and OpenTelemetry traces, the
mental model is the same.

## Where to go next

- [**Quickstart**](quickstart.md) — install LazyBridge and run your first
  agent in five minutes.
- [**Ecosystem**](ecosystem.md) — the three packages (lazybridge core,
  lazytools capabilities, lazypulse always-on) and which one you need.
- [**Concepts → Mental model**](concepts/mental-model.md) — Agent =
  Engine + Tools + State, the only decomposition you need.
- [**Concepts → Everything is a tool**](concepts/everything-is-a-tool.md) —
  the composition rule that holds the framework together.
- [**Concepts → Progressive complexity**](concepts/progressive-complexity.md) —
  the twelve rungs from one-line agent to checkpointed pipeline.
- [**Concepts → Canonical vs sugar**](concepts/canonical-vs-sugar.md) —
  every factory function and shortcut LazyBridge ships, with its
  canonical equivalent and any subtle differences.
- **Guides → Basic** — one focused page per Day-1 concept:
  [Agent](guides/basic/agent.md), [Tool](guides/basic/tool.md),
  [Envelope](guides/basic/envelope.md),
  [Native tools](guides/basic/native-tools.md).
- **Guides → Mid** — twelve pages on real-app concerns:
  [Memory](guides/mid/memory.md), [Store](guides/mid/store.md),
  [Session](guides/mid/session.md), [Guards](guides/mid/guards.md),
  [verify=](guides/mid/verify.md), [Chain](guides/mid/chain.md),
  [Parallel](guides/mid/parallel.md), [As tool](guides/mid/as-tool.md),
  [HumanEngine](guides/mid/human-engine.md), [MCP](guides/mid/mcp.md),
  [Multimodal](guides/mid/multimodal.md), [Evals](guides/mid/evals.md).
- **Guides → Full** — nine pages on production pipelines:
  [Plan](guides/full/plan.md), [Step](guides/full/step.md),
  [Sentinels](guides/full/sentinels.md), [Routing](guides/full/routing.md),
  [Parallel plan steps](guides/full/parallel-plan-steps.md),
  [Checkpoint & resume](guides/full/checkpoint.md),
  [Exporters](guides/full/exporters.md),
  [GraphSchema](guides/full/graph-schema.md),
  [SupervisorEngine](guides/full/supervisor.md).
- **Guides → Advanced** — seven pages on the extension surface:
  [Engine protocol](guides/advanced/engine-protocol.md),
  [BaseProvider](guides/advanced/base-provider.md),
  [Providers catalogue](guides/advanced/providers.md),
  [External tool gateway](guides/advanced/external-tools.md),
  [Plan serialization](guides/advanced/plan-serialize.md),
  [OpenTelemetry](guides/advanced/otel.md),
  [Visualizer](guides/advanced/visualizer.md).
- [**Recipes**](recipes/index.md) — runnable examples from the
  `examples/` directory, embedded verbatim.
- [**Decisions**](decisions/index.md) — nine "which one do I
  use?" decision trees with quick-reference tables.
- [**Reference**](reference/index.md) — auto-generated API surface
  for every public symbol, organised by category.
- [**Errors**](errors.md) — cause → diagnosis → fix table for
  every framework exception and `Envelope.error.type` value.
- [**For LLM assistants**](for-llms/index.md) — Claude Skill install,
  `/llms.txt` index, and `/llms-full.txt` corpus dump.

## Design principles

- **Provider freedom.** Switch models or providers without rewriting
  your architecture.
- **Everything is a tool.** Functions, agents, plans, pipelines, MCP
  servers, and external systems all compose through the same primitive.
- **Zero boilerplate.** No duplicated function definitions, no manual
  JSON schema translation, no orchestration glue you have to maintain.
- **Progressive complexity.** Simple use cases stay simple. Complex
  workflows are possible without changing the core mental model.
- **Designed for humans and LLMs.** Code that's readable to a
  reviewer is also learnable to an assistant that writes the next
  patch.
- **Determinism when you need it.** Drop down from autonomous LLM
  loops to a typed, validated `Plan` whenever auditability or repeat
  cost matters.
- **Composability over monoliths.** Large systems emerge from small,
  specialised components — not one overloaded prompt.
- **Observable and debuggable.** Sessions, exporters, the Visualizer,
  and OpenTelemetry mean you can always see what happened and why.

LazyBridge is meant to feel like a bridge, not a cage.

## Maturity

LazyBridge 0.7.9 is on PyPI as **Alpha** (`Development Status :: 3` in
PyPI metadata, `lazybridge.__stability__ = "alpha"`).  The public API
is stable enough that breaking changes go through the migration guides
under [`docs/migrations/`](migrations/0.7-to-0.79.md), but production
hardening is uneven by subsystem.  The labels below describe the
state of each feature area in this release.

| Subsystem | Status | Notes |
|---|---|---|
| `Agent`, `LLMEngine`, `Tool`, `Envelope` | **Stable** | Public surface, exercised by every test path. |
| `Plan`, `Step`, sentinels, routing | **Stable** | Compiler validates at construction; serialisation supported. |
| `Memory`, `Store` (in-memory + SQLite) | **Stable** | API frozen; encrypted store adapter is also stable. |
| `Session`, `EventLog`, exporters, `GraphSchema` | **Stable** | Default secret redaction enabled (`redact_secrets`). |
| Provider adapters (Anthropic / OpenAI / Google / DeepSeek / LiteLLM / LM Studio) | **Stable** | Adapters are stable; model/price tables drift with provider releases. |
| MCP server integration | **Alpha** | Deny-by-default tool allow-list; spec coverage tracked in changelog. |
| Native tools (`NativeTool`) | **Alpha** | Provider-hosted capabilities (web search, code interpreter). Surface area changes when providers add new tools. |
| `Checkpoint` / `resume`  | **Alpha** | Internal-state atomic across parallel `Plan` bands; *external* side-effect rollback is not implemented (see [Parallel plan steps](guides/full/parallel-plan-steps.md)). |
| Guardrails (`Guard`, `ContentGuard`, `LLMGuard`, `GuardChain`) | **Alpha** | Behaviour is stable; default rule libraries are still growing. |
| `HumanEngine`, `SupervisorEngine` | **Alpha** | Public API stable; UX polish continues. |
| Evals (`lazybridge.ext.evals`) | **Experimental** | Scoring helpers are stable; the runner API may consolidate before 1.0. |
| Visualizer (`lazybridge.ext.viz`) | **Experimental** | Useful for debugging; not part of the runtime path. |
| Provider model fallback chains (`_FALLBACKS`) | **Planned** | Data tables exist; the retry path that consumes them is not implemented yet. |
| Automatic PII redaction | **Planned** | Default redactor masks credential shapes only.  Compose your own for emails / phone numbers / SSNs. |

The roadmap toward a 1.0 stable surface is to lift each Alpha row to
Stable, ship the Planned items where the design is settled, and either
implement or remove the experimental modules based on user feedback.
