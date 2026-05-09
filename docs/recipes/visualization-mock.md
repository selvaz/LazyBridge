# Visualization mock

The same Visualizer UI as [Live visualization](live-visualization.md),
but driven by a synthetic event stream — no LLM calls, no provider
keys required. Useful for demos, screen recordings, or for
exercising the UI without spending tokens.

Demonstrates a complex topology: planner → parallel researchers
(each with a nested sub-researcher) → merger → writer, with store
writes and explicit graph registrations along the way.

## Source

```python
--8<-- "examples/viz_mock_demo.py"
```

## Walkthrough

- **No real agents.** The script registers nodes / edges directly
  on `session.graph` and emits events via `session.emit(...)` to
  simulate a multi-agent pipeline. The Visualizer reads the same
  surfaces it does in live mode.
- **Nested topology.** Each parallel researcher has a sub-researcher
  in its own `tools=[...]` — the graph has two levels of
  `as_tool` edges. Useful for stress-testing the renderer.
- **Store writes.** The mock pushes values into the `Store` to
  exercise the side-panel store viewer that highlights writes as
  they happen.

## Variations

- Drive the mock from a recorded production trace (load events
  from a JSONL file, replay them through `session.emit`) — useful
  for triaging an issue without re-running the production
  pipeline.
- Slow the event timeline down (`time.sleep(0.5)` between emits)
  for screen-recording demos where you want viewers to follow
  along.

## See also

- [Live visualization](live-visualization.md) — the real-LLM
  sibling.
- [Visualizer](../guides/advanced/visualizer.md) — the deep
  reference, including the `EventHub` / `HubExporter` surface
  for custom UIs.
- [GraphSchema](../guides/full/graph-schema.md) — the topology
  representation the mock builds explicitly.
