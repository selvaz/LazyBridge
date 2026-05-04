# report_builder — Usage Reference

## Setup

```python
from lazybridge import Agent
from lazybridge.ext.report_builder import report_tools

agent = Agent("anthropic", tools=report_tools("./reports"))
```

`report_tools(output_dir)` returns a single tool: `generate_report`.

## generate_report — Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `markdown_path` | `str` | yes | Path to the `.md` narrative file |
| `title` | `str` | yes | Report title (browser tab + document heading) |
| `theme` | `str` | no | `executive` \| `financial` \| `technical` \| `research` (default: `executive`) |
| `charts` | `list[dict]` | no | List of chart references (see below) |
| `output_filename` | `str` | no | Output filename (default: `report.html`) |

### Chart reference dict

```python
{
    "path":  "./output/charts/revenue.png",   # path to existing PNG
    "title": "Revenue by Quarter",             # caption under the chart
    "name":  "Revenue by Quarter",             # used for heading-based auto-placement
}
```

The `name` field is matched against `<h1>`/`<h2>`/`<h3>` headings in the rendered
Markdown using word-overlap scoring.  The chart is inserted immediately after the
best-matching heading.  If `name` is omitted, `title` is used for matching.
Charts with no heading match are appended at the end of the document.

## Return value

On success:
```python
{
    "html_path":       "/abs/path/to/report.html",
    "title":           "Q1 2026 Report",
    "charts_embedded": 3,
    "theme":           "executive",
}
```

On failure:
```python
{"error": True, "type": "FileNotFoundError", "message": "..."}
```

## Full example

```python
result = agent.tools["generate_report"](
    markdown_path="./output/analysis.md",
    title="Q1 2026 Performance Review",
    theme="financial",
    charts=[
        {"path": "./output/charts/revenue_bar.png",
         "title": "Revenue by Quarter",
         "name": "Revenue"},
        {"path": "./output/charts/segment_pie.png",
         "title": "Segment Mix",
         "name": "Segment"},
    ],
    output_filename="q1_2026.html",
)
print(result["html_path"])
```
