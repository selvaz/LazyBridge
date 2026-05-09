# Daily news

A multi-region news pipeline: three region pipelines (each a
discovery step + N parallel article writers) run concurrently,
their outputs feed region assemblers, a final orchestrator
synthesises the cross-region report, and an HTML designer
formats the output. Configurable depth (brief / standard / deep).

Uses `Store(db="...")`, `Session`, multiple providers (Anthropic,
OpenAI, DeepSeek, Google), `native_tools=[NativeTool.WEB_SEARCH]`,
custom tools (`fetch_image`, `search_wikimedia`), nested `Plan`s,
`from_parallel_all`, and `report_builder` — the closest the
example corpus comes to a real production pipeline.

## Source

```python
--8<-- "examples/daily_news_report.py"
```

## Walkthrough

- **Nested `Plan`s.** Each region's pipeline is itself a `Plan`
  (discovery → N parallel writers → assembler) wrapped in an
  `Agent`. The top-level orchestrator's plan dispatches three of
  these region agents in parallel, then runs a final synth +
  HTML-designer pair.
- **Multi-provider routing.** Different roles use different
  providers — Anthropic for synthesis, OpenAI for image fetching,
  DeepSeek for cheap drafting, Google for grounding. The provider
  is per-engine; the rest of the agent shape is uniform.
- **`native_tools=[NativeTool.WEB_SEARCH]`** opts into provider-
  hosted web search (Anthropic) for the discovery step. Custom
  Python tools (`fetch_image`, `search_wikimedia`) coexist with
  native ones.
- **`Store(db="...")`** persists every step's output — combined
  with `Session(db="events.sqlite")` you get a complete audit
  trail you can replay through the [Visualizer](live-visualization.md).
- **Configurable depth** (`brief` / `standard` / `deep`) drives
  the number of writer branches per region. The same pipeline
  shape scales from "tiny demo" to "real production" by changing
  one argument.

## Variations

- Disable the visualizer hook for headless runs (CI, cron) — pass
  `auto_open=False` to the `Visualizer` constructor or omit it
  entirely.
- Swap the HTML designer for a Quarto exporter (the
  `report_builder` exporter surface supports both) for academic-
  paper output.
- Cap concurrency per region with
  `Plan(...).run_many(tasks, concurrency=N)` if you're rate-
  limited at the provider.

## See also

- [Parallel report](parallel-report.md) — the simpler sibling
  that introduces the fragment / report pattern.
- [Parallel plan steps](../guides/full/parallel-plan-steps.md) —
  the band atomicity rules this pipeline relies on.
- [Native tools](../guides/basic/native-tools.md) —
  `WEB_SEARCH` and the per-provider compatibility matrix.
- [Live visualization](live-visualization.md) — pair with the
  Visualizer to watch the multi-region pipeline as it runs.
