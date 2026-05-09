# Researcher → reporter

Two agents in sequence: a researcher gathers facts, a reporter
turns them into a markdown brief. The handoff is text — the
researcher's output `Envelope.text()` becomes the reporter's task.
The LazyBridge equivalent of CrewAI's `Process.sequential` crew,
without `@CrewBase`, decorators, or YAML.

## Source

```python
--8<-- "examples/crewai/02_research_and_report.py"
```

## Walkthrough

- **`Agent.chain(researcher, reporter)`** is sugar for the canonical
  `Agent(engine=Plan(Step(target=researcher, name=researcher.name),
  Step(target=reporter, name=reporter.name)), name="chain")` form. Use
  the sugar for purely linear handoffs; reach for the explicit
  `Plan` form when you need typed handoffs, routing, or
  checkpoints.
- **`crew(f"Topic: {topic}")`** is the canonical sync call —
  returns the final agent's `Envelope`. `.text()` extracts the
  markdown report.
- **Each builder function (`build_researcher` / `build_reporting_analyst`)**
  is a plain Python factory; the role / goal / backstory go in the
  `system` prompt instead of an external YAML file.

## Variations

- Replace the chain with a `Plan` for typed Pydantic handoff:
  `Step(researcher, output=ResearchPayload)` then
  `Step(writer, context=from_step("researcher"))` — the writer sees
  the typed payload.
- Add a `verify=judge` to the reporter to gate the final output on
  policy (length, format, citation presence).
- For a third agent in the pipeline — fact-checker, copy-editor —
  just append to `Agent.chain(...)` or to the underlying `Plan`.

## See also

- [Chain](../guides/mid/chain.md) — sugar for sequential pipelines.
- [Plan](../guides/full/plan.md) — the canonical orchestration
  primitive, for typed handoffs and conditional routing.
- [Researcher (single agent)](researcher-single.md) — the previous
  rung.
- [Supervisor pattern](supervisor-pattern.md) — the next rung:
  hierarchical dispatch, where a supervisor decides which agent
  to call.
