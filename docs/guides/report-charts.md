# Chart contract

Charts in the fragment-based reporting pipeline are emitted as **JSON
specs** — never as Python code.  Two engines are supported through
`append_chart` and the `ChartSpec` schema:

| Engine        | Default? | Why                                     |
|---------------|----------|------------------------------------------|
| `vega-lite`   | yes      | Pure-Rust rasterizer (`vl-convert-python`) — no Node, no Chrome. |
| `plotly`      | opt-in   | Familiar to many models; raster path needs Chrome via kaleido. |

This is a deliberate constraint: JSON is validatable, sandboxes-free,
serialisable through the bus, and survives a crash.  No matplotlib, no
seaborn, no LLM-emitted code that we'd have to `exec`.

## Vega-Lite (default)

### Spec shape

```python
from lazybridge.external_tools.report_builder import ChartSpec

ChartSpec(
    engine="vega-lite",
    title="Quarterly revenue",
    spec={
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": "bar",
        "encoding": {
            "x": {"field": "quarter", "type": "ordinal"},
            "y": {"field": "revenue", "type": "quantitative"},
        },
    },
    data=[
        {"quarter": "Q1", "revenue": 12.4},
        {"quarter": "Q2", "revenue": 18.1},
        {"quarter": "Q3", "revenue": 21.0},
        {"quarter": "Q4", "revenue": 27.5},
    ],
)
```

The validator accepts any spec with `mark`, `layer`, `hconcat`,
`vconcat`, `facet`, or `repeat` at the top level.  Anything else fails
fast with a helpful error before the bus is touched.

### Render outputs

| Output | How                                              | When emitted                           |
|--------|--------------------------------------------------|-----------------------------------------|
| HTML   | `<div>` + vega-embed via CDN                     | Always (interactive, with tooltips).    |
| SVG    | `vl_convert.vegalite_to_svg(spec)`               | When the exporter requests raster.      |
| PNG    | `vl_convert.vegalite_to_png(spec, scale=2.0)`    | Best-effort; SVG is the lossless fallback. |

The interactive HTML embed loads vega/vega-lite/vega-embed scripts from
`cdn.jsdelivr.net`.  Multiple charts on the same page share the same
`<script>` requests (browser cache).  Each `<div>` gets a unique id so
embeds don't collide.

### Inline `data` splice

When you supply `data=`, it overrides `spec.data.values`.  This lets the
same spec be reused with different data rows — handy when an LLM
authors a "spec template" and the agent that calls `append_chart` plugs
the numbers in.

```python
TEMPLATE = {
    "mark": "bar",
    "encoding": {
        "x": {"field": "label", "type": "ordinal"},
        "y": {"field": "value", "type": "quantitative"},
    },
}
bus.append(Fragment(
    kind="chart",
    chart=ChartSpec(spec=TEMPLATE, data=quarterly_revenue, title="Revenue"),
))
```

### Common mark types

`bar` · `line` · `area` · `point` · `arc` (pie) · `rect` (heatmap) ·
`circle` · `tick` · `rule` · `text` · `boxplot` · `errorband` · `image`.
Layered specs combine multiple marks; concat / hconcat / vconcat compose
multiple charts.

### Themes and styling

Vega-Lite has its own `config.view`, `config.style`, and theme presets
(`config.style.bar.fill = "..."`).  The Quarto exporter does not
override these — what you put in the spec is what you get.  For a
consistent look across all charts in the report, define a default config
in your spec template.

## Plotly (opt-in)

### Spec shape

```python
ChartSpec(
    engine="plotly",
    title="Latency p95 over time",
    spec={
        "data": [
            {"type": "scatter", "mode": "lines+markers"},
        ],
        "layout": {"title": "p95 latency"},
    },
    data=[
        {"x": "2026-01-01", "y": 142},
        {"x": "2026-02-01", "y": 138},
        {"x": "2026-03-01", "y": 121},
    ],
)
```

The validator requires a `data` list at the top level.

### Inline `data` splice

The convenience splice fills `x` and `y` of the **first trace** from the
**first two columns** of your inline rows.  For multi-trace specs
(scatter + line + area in one figure), populate `spec.data` directly and
omit the inline `data=` argument.

### Render outputs

| Output | How                                                        | Caveats |
|--------|------------------------------------------------------------|---------|
| HTML   | `plotly.io.to_html(fig, include_plotlyjs="cdn", full_html=False)` | CDN script tag injected once per page. |
| PNG    | `plotly.io.to_image(fig, format="png")` via kaleido        | Requires kaleido + system Chrome. Best-effort. |
| SVG    | `plotly.io.to_image(fig, format="svg")` via kaleido        | Same caveat. |

When kaleido isn't installed (the default — we deliberately don't pull
it transitively), the raster path silently no-ops and the exporter falls
back to embedding the interactive HTML.  PDF / DOCX output is then
degraded for that fragment, but the rest of the report is unaffected.

To enable raster:

```bash
pip install kaleido
# Linux: ensure a Chromium build is on PATH or set:
#   PLOTLY_KALEIDO_CHROMIUM=/usr/bin/chromium
```

## When to use which

| Scenario                                              | Engine      |
|-------------------------------------------------------|-------------|
| Headless server, no Chrome, deterministic raster      | Vega-Lite   |
| LLM is more fluent in Plotly (it appears in pretraining) | Plotly      |
| Many small charts on the same page (faceted bars)     | Vega-Lite (smaller HTML) |
| 3D, geographic, or specialty traces (sankey, funnel)  | Plotly      |
| Shared style across the whole report                  | Vega-Lite via shared `config` |

## Common gotchas

* **Don't emit Python code.**  The chart contract is JSON.  If a power
  user really needs matplotlib, they can render the PNG in their own Step
  and attach it as a `text` fragment with an `<img>` tag, or use the
  legacy single-shot `generate_report()` flow which accepts pre-rendered
  PNG paths.
* **Keep `data` arrays small.**  10k+ rows in a JSON spec inflate the
  HTML output and choke the in-memory bus.  Aggregate before charting.
* **Set the `title` field.**  Both the renderer and the Pandoc figure
  caption use it.  An empty title produces an unlabelled figure that
  isn't searchable in TOCs.
* **Vega-Lite `$schema` is optional but helpful.**  Some tools key
  validation off it.  Including
  `"$schema": "https://vega.github.io/schema/vega-lite/v5.json"` makes
  the spec self-describing.
* **Plotly + Quarto: PDF rasterization.**  Quarto pre-rasterises Plotly
  blocks for non-HTML formats *only when kaleido is present*.  Without
  kaleido, PDFs and DOCX show "(figure rendered in HTML only)" — install
  kaleido or switch the chart to Vega-Lite.

## Programmatic chart rendering

The render path is exposed for testing and custom exporters:

```python
from lazybridge.external_tools.report_builder import ChartSpec
from lazybridge.external_tools.report_builder.charts import render_chart

spec = ChartSpec(spec={"mark": "bar", "encoding": {}})
rendered = render_chart(spec, raster=True)
print(rendered.html[:200])
print(len(rendered.svg or ""))
print(len(rendered.png_bytes or b""))
```

`render_chart` dispatches to the right engine, returns a `RenderedChart`
with `html` / `svg` / `png_bytes`.  Invalid specs raise `ValueError`.

## See also

- [Fragment schema](report-fragments.md) — `ChartSpec` field reference
- [Exporters](report-exporters.md) — how charts render in each format
- [Vega-Lite docs](https://vega.github.io/vega-lite/docs/)
- [Plotly Python docs](https://plotly.com/python/)
