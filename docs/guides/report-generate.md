# Single-shot `generate_report`

`report_tools(output_dir)` returns a single LLM-callable `Tool` —
`generate_report` — that assembles a complete HTML (and optionally PDF)
report from a Markdown file + chart PNGs, or from a typed list of
content sections.  This is the original API and is unchanged by the
fragment-based subsystem.

When to pick this over the [fragment workflow](report-builder.md#quick-start-parallel-fragments):

* One agent does all the work end-to-end.
* You already have a Markdown file from another tool.
* You don't need DOCX, Reveal.js, citations, or per-fragment provenance.

## Construction

```python
from lazybridge import Agent
from lazybridge.external_tools.report_builder import report_tools

agent = Agent(model="anthropic:claude-sonnet-4-6", tools=report_tools("./out"))
```

`output_dir` is the directory the tool writes outputs into; created on
first call if missing.

## Tool reference: `generate_report`

```python
generate_report(
    title: str,
    theme: str = "executive",
    template: str = "default",
    sections: list[dict] | None = None,
    markdown_path: str | None = None,
    charts: list[dict] | None = None,
    output_filename: str = "report.html",
    output_format: str = "html",
) -> dict
```

### Parameters

| Argument          | Type            | Notes |
|-------------------|-----------------|-------|
| `title`           | str             | Browser tab + h1 heading. |
| `theme`           | str             | One of `"executive"`, `"financial"`, `"technical"`, `"research"`. |
| `template`        | str             | One of `"default"`, `"executive_summary"`, `"deep_dive"`, `"data_snapshot"`. |
| `sections`        | list[dict] \| None | Typed content blocks (see below).  Mutually exclusive with `markdown_path`. |
| `markdown_path`   | str \| None     | Path to an existing `.md` file. |
| `charts`          | list[dict] \| None | Chart refs for the markdown_path flow only.  Each: `{"path": str, "title": str, "name": str}`. |
| `output_filename` | str             | Output basename — `.html` is appended if missing. |
| `output_format`   | str             | `"html"`, `"pdf"`, or `"both"`.  PDF requires `lazybridge[pdf]`. |

### Returns

```python
{
    "title": str,
    "theme": str,
    "template": str,
    "charts_embedded": int,
    "html_path": str | None,
    "pdf_path": str | None,
}
```

On error: `{"error": True, "type": str, "message": str}`.  Exceptions
are caught and wrapped — the LLM sees an error dict, not a stack trace.

## Two input flows

### `sections=` flow

Pass typed content blocks directly.  No file required.

```python
sections = [
    {"type": "text", "heading": "Executive Summary",
     "body": "**Q1 was strong.**  Revenue grew 18% YoY..."},
    {"type": "table", "caption": "Quarterly revenue (USD M)",
     "headers": ["Quarter", "Revenue", "YoY"],
     "rows": [["Q1", "12.4", "+18%"], ["Q2", "18.1", "+22%"]]},
    {"type": "chart", "heading": "Revenue trend",
     "path": "./charts/revenue.png", "title": "Revenue 2024-2026"},
]

agent("""
Call generate_report with title='Q1 2026 Review', theme='executive',
template='executive_summary', and sections=<the JSON blocks above>.
""")
```

Section types:

| `type`     | Required fields              | Optional |
|------------|------------------------------|----------|
| `"text"`   | `body` (Markdown)            | `heading` |
| `"chart"`  | `path` (PNG file), `title`   | `heading` |
| `"table"`  | `headers`, `rows`            | `caption` |

Sections render in declaration order — no auto-placement.

### `markdown_path=` flow

Pass a path to an existing `.md` file plus optional chart refs.

```python
agent("""
Read /tmp/analysis.md, generate the report:
- title: 'Q1 2026 Review'
- markdown_path: '/tmp/analysis.md'
- charts: [
    {"path": "./charts/revenue.png", "title": "Revenue 2024-2026", "name": "Revenue"},
    {"path": "./charts/segments.png", "title": "Segment mix", "name": "Segment"}
  ]
""")
```

`charts` are auto-placed: each chart's `name` is matched against the
section headings in the rendered Markdown via word-overlap scoring.
Best match wins; unmatched charts append at the end.

## Themes

| Theme       | Style                                   | Use case                              |
|-------------|------------------------------------------|---------------------------------------|
| `executive` | Dark-blue accent, Inter font, rounded card | Board reports, strategy documents     |
| `financial` | Teal accent, tabular numbers             | KPI dashboards, P&L reviews           |
| `technical` | Dark mode, cyan accent, mono headings    | Engineering reports, incident reviews |
| `research`  | Serif (Georgia), rust accent, academic   | White papers, market research         |

CSS variables are consistent across themes — `--accent`, `--bg`,
`--surface`, `--text`, `--muted`, `--border` — so a custom theme can be
added by writing one `.css` file with the same vars.

## Templates

| Template            | Layout                                          |
|---------------------|--------------------------------------------------|
| `default`           | Single column, no chrome.                        |
| `executive_summary` | Header bar with title + meta chips + footer.    |
| `deep_dive`         | Sticky TOC sidebar (h2 + h3) + full-width content. |
| `data_snapshot`     | Chart-prominent — figures full width with large captions. |

## PDF output

```bash
pip install lazybridge[pdf]
```

```python
agent("""
... same as above, with output_format='both' to also produce PDF.
""")
```

Output: `report.html` + `report.pdf` (same basename, `.pdf` extension).

## Self-contained HTML

All chart PNGs are base64-encoded and embedded as `data:` URIs.  The
output `.html` carries everything it needs to render — copy it
anywhere, no asset directory required.

```bash
$ wc -l out/report.html         # ~5000 lines, ~1MB for a 4-chart report
$ open out/report.html          # works offline
```

## Errors

* `Unknown theme` — invalid `theme` argument.
* `Unknown template` — invalid `template` argument.
* `Unknown output_format` — must be `"html"`, `"pdf"`, or `"both"`.
* `Provide either sections or markdown_path` — at least one required.
* `Markdown file not found` — `markdown_path` doesn't exist.
* `Chart image not found` — chart `path` doesn't exist.
* `sections[i] is invalid` — typed-block validation failed; the message
  carries the pydantic error.
* `weasyprint is required for PDF output` — install `lazybridge[pdf]`.

All errors come back through the standard error dict shape so the LLM
can self-correct.

## Migrating to the fragment workflow

If you outgrow the single-shot tool — multi-agent contributions, real
citations, multi-format output — the fragment workflow drops in
incrementally.  See [the fragment-based reporting guide](report-builder.md#why-two-apis)
for when to switch.

## See also

- [Report builder overview](report-builder.md)
- [Decision: which report shape?](../decisions/report-shape.md)
