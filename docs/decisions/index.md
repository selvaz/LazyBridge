# Decisions

Short reference pages, one per recurring "which one do I use?"
question. Each page has a decision tree, a quick-reference table,
and pointers to the deep guides for the chosen option.

## Picking your starting point

- [Which tier?](pick-tier.md) — Basic, Mid, Full, or Advanced.

## Inside an Agent

- [Return type](return-type.md) — `.text()` vs `.payload` vs
  `.metadata`.
- [State layer](state-layer.md) — `Memory`, `Store`, or `sources=`.

## Composing Agents

- [Composition](composition.md) — `Agent.chain` vs
  `Agent.parallel` vs `Agent(tools=[...])` vs `Plan`.
- [Parallelism](parallelism.md) — automatic (LLM-decided) vs
  declared.

## Human / verifier placement

- [HumanEngine vs SupervisorEngine](human-engine-vs-supervisor.md)
- [`verify=` placement](verify-placement.md) — Agent / tool /
  Plan step.

## Production-grade Plan

- [Checkpoint & resume](checkpoint.md) — when is the storage
  overhead worth it?

## Extension surface

- [Do I need Advanced?](need-advanced.md) — smell test for the
  framework-author tier.
