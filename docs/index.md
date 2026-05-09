# LazyBridge

A zero-boilerplate, multi-provider Python framework for composing LLMs, plain
functions, deterministic plans, humans, and external tools under one uniform
model: **everything is a tool.**

```python
from lazybridge import Agent, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
)
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
- **Guides → Mid / Full / Advanced** *(coming next)* — Memory,
  Store, Session, Plan, sentinels, routing, checkpoints,
  human-in-the-loop, OpenTelemetry, custom engines and providers.
- **Recipes** *(coming next)* — runnable end-to-end examples lifted from
  `examples/`.
- **For LLM assistants** *(coming next)* — install the LazyBridge Claude
  Skill, point your AI tooling at `/llms.txt`, or connect the docs MCP
  server.

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
