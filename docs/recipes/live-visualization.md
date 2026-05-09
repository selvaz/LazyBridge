# Live visualization

A 3-agent chain (researcher → analyst → writer) wrapped in a
`Visualizer` context manager. The browser tab opens automatically;
pulses travel along graph edges as agents call tools, the
inspector lets you click into every event, the store viewer
highlights writes as they happen.

The same UI replays a finished session via
`Visualizer.replay(db="...").open()`.

## Source

```python
--8<-- "examples/viz_demo.py"
```

## Walkthrough

- **`with Visualizer(sess) as viz`** — context-manager pattern is
  what the recipe shows. The HTTP server starts on `__enter__`,
  shuts down on `__exit__`. The browser tab survives the boundary;
  close it yourself.
- **`session=sess` on every agent** — the live `GraphSchema` is
  populated from the agents that register with the session. An
  agent without `session=sess` is invisible in the topology even
  if it runs.
- **`viz.url`** is the bound URL (`127.0.0.1:<ephemeral>` by
  default). Print it before the pipeline runs so you can switch
  to the browser before tools start firing.

## Variations

- For a fixed port (e.g. to share with a teammate over a tunnel),
  pass `port=8765` to `Visualizer(sess, port=8765)`.
- For headless / CI runs that should NOT open a browser, pass
  `auto_open=False`.
- For a recorded run replayed at half speed:
  `Visualizer.replay(db="demo.db", speed=0.5).open()`.

## See also

- [Visualizer](../guides/advanced/visualizer.md) — the deep
  reference for live + replay modes, control endpoints, and
  custom-UI hooks.
- [Visualization mock](visualization-mock.md) — the same UI
  driven by a synthetic event stream (no LLM calls).
- [Session](../guides/mid/session.md) — the bus the Visualizer
  reads from; covers `db=`, `batched=`, exporters.
