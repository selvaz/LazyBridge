# Pick your tier

> **Where do I start: Basic, Mid, Full, or Advanced?**

Start as low as possible. Tiers are additive — moving up never
requires changing the code you already wrote.

## Decision tree

```text
Single agent, one call, maybe with one tool?
    → Basic.

Need conversation memory, observability, guardrails, or simple
chain / parallel composition?
    → Mid.

Declared multi-step workflow with typed hand-offs, routing,
crash-resume, or per-step verifiers?
    → Full.

Writing framework code — new provider, new engine,
cross-process Plan serialisation?
    → Advanced.
```

## Quick reference

| You need | Start at | Key surface |
|---|---|---|
| One agent, one tool, one call | **Basic** | `Agent(engine=LLMEngine("…"), tools=[…])` + `Envelope` |
| Memory, Store, Session, Guards, `chain` / `parallel` / `as_tool`, MCP, Evals, HumanEngine, `verify=` | **Mid** | `Memory`, `Store`, `Session`, `Agent.chain`, `Agent.parallel`, MCP, `verify=`, `HumanEngine` |
| Plan + Step + sentinels, routing, parallel bands, checkpoint / resume, exporters, GraphSchema, `SupervisorEngine` | **Full** | `Plan`, `Step`, `from_*` sentinels, `Step(routes=…)`, `Step(parallel=True)`, `Session(exporters=[…])` |
| Custom engine, custom provider, Plan persistence, OTel deep, Visualizer | **Advanced** | `BaseProvider`, `Engine` protocol, `Plan.to_dict`, `OTelExporter`, `Visualizer` |

## Notes

- **Tiers are additive.** Adding a `Memory` to a Basic agent
  doesn't require restructuring; you just pass it as a kwarg.
  Same for `Session`, `output=`, `verify=`, `tools=[...]`.
- **Don't reach up too early.** A Plan is overkill for a 3-step
  text pipeline (use `Agent.chain`); `verify=` is overkill for a
  policy a regex `Guard` already enforces; a custom engine is
  overkill if `LLMEngine(system="...", max_turns=N)` covers
  what you need.
- **Advanced is framework authorship**, not application
  development. If you're tweaking prompts / swapping models /
  building pipelines, you're in Basic / Mid / Full.

## See also

- [Mental model](../concepts/mental-model.md) — Engine + Tools +
  State, the decomposition every tier shares.
- [Progressive complexity](../concepts/progressive-complexity.md)
  — the twelve-rung ladder from one-liner to checkpointed
  pipeline.
- [Canonical vs sugar](../concepts/canonical-vs-sugar.md) — the
  full table of factory shortcuts mapped back to canonical
  forms.
