# Chart contract for LLM agents

Two engines are supported through `append_chart`:

| Engine       | When to pick it                                     |
|--------------|-----------------------------------------------------|
| `vega-lite`  | Default.  Pure-Rust rasterizer, no Chrome, deterministic. |
| `plotly`     | Opt-in.  Familiar to many models, raster path needs Chrome. |

Charts are emitted as JSON specs — never Python code.  This keeps the
contract validatable, sandboxes-free, and serialisable through the bus.

## Vega-Lite

```json
{
  "engine": "vega-lite",
  "title": "Quarterly revenue",
  "spec": {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "mark": "bar",
    "encoding": {
      "x": {"field": "quarter", "type": "ordinal"},
      "y": {"field": "revenue", "type": "quantitative"}
    }
  },
  "data": [
    {"quarter": "Q1", "revenue": 12.4},
    {"quarter": "Q2", "revenue": 18.1},
    {"quarter": "Q3", "revenue": 21.0},
    {"quarter": "Q4", "revenue": 27.5}
  ]
}
```

Notes:

* `data` is optional — when omitted, Vega-Lite uses whatever
  `spec.data.values` carries.  When present, it overrides.
* The renderer embeds an interactive `<div>` in HTML (vega-embed via CDN)
  and rasterises to SVG / PNG via `vl-convert-python` for PDF / DOCX.
* All standard mark types work: `bar`, `line`, `area`, `point`, `arc`
  (pie), `boxplot`, `circle`, `tick`, `rule`, `text`, plus layered specs.

## Plotly

```json
{
  "engine": "plotly",
  "title": "Latency p95 over time",
  "spec": {
    "data": [
      {"type": "scatter", "mode": "lines+markers"}
    ],
    "layout": {"title": "p95 latency"}
  },
  "data": [
    {"x": "2026-01-01", "y": 142},
    {"x": "2026-02-01", "y": 138},
    {"x": "2026-03-01", "y": 121}
  ]
}
```

Notes:

* The convenience `data` splice fills `x` and `y` of the first trace from
  the first two columns of your rows.  For complex multi-trace charts
  populate `spec.data` directly and omit `data`.
* HTML embed uses the Plotly.js CDN; PDF / DOCX rasterisation requires
  `kaleido` which in turn requires a system Chrome.  When kaleido fails
  the renderer keeps the interactive HTML and skips the raster — output
  is degraded but never broken.

## Common gotchas

* Don't emit Python code for charts (matplotlib, seaborn).  We don't run
  `exec` on LLM output.  If a power user needs them, they can render the
  PNG in their own Step and attach it as a `text` fragment with an `<img>`
  tag.
* Keep `data` arrays small.  10k rows in a JSON spec inflates report size
  and chokes the in-memory bus.  Aggregate before charting.
* Set the `title` field — both the renderer and the Pandoc figure caption
  use it.
