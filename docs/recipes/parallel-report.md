# Parallel report

A fragment-based reporting pipeline: three researchers run in
parallel, each appending fragments to a shared `FragmentBus`; an
exec-summary agent reads the bus and produces a final summary;
the result is exported to HTML / PDF / Reveal.js. Runs without an
LLM key (uses a mock provider) so it's safe to iterate on locally.

Demonstrates the `lazybridge.external_tools.report_builder` surface
end-to-end.

## Source

```python
--8<-- "examples/parallel_report_pipeline.py"
```

## Walkthrough

- **`Plan(Step(..., parallel=True), Step(..., parallel=True), Step(..., parallel=True), Step("synth", ...))`**
  — three concurrent branches followed by an implicit join. The
  synth step reads all branches via `from_parallel_all(...)` or
  via the `FragmentBus`.
- **`FragmentBus`** is a shared scratchpad: branches append
  fragments; downstream steps read them in order. Cleaner than
  threading text through `from_parallel_all` when many short
  fragments come from many branches.
- **`OutlineAssembler`** folds fragments into a structured
  multi-section report. The exporter converts to HTML / PDF /
  Reveal.js / Quarto without re-running the agents.

## Variations

- Replace the mock provider with a real one to exercise live
  research — the rest of the pipeline is unchanged.
- Add a fourth branch (`Step(parallel=True, ...)`) — concurrent
  bands grow horizontally without restructuring downstream
  steps.
- Persist the fragments to a `Store` for replay / audit; the
  `BlackboardAssembler` reads from a Store rather than the
  in-memory bus.

## See also

- [Parallel plan steps](../guides/full/parallel-plan-steps.md) —
  the band semantics underlying this recipe.
- [Daily news](daily-news.md) — the production-shape mega
  example that uses the same fragment / report pattern at
  larger scale.
- [Sentinels](../guides/full/sentinels.md) —
  `from_parallel_all("first_branch")` for the join step's
  context.
