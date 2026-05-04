# report_builder — Overview

`lazybridge.ext.report_builder` assembles self-contained HTML reports from
Markdown files and pre-generated chart PNG images.

## When to use

Call `generate_report` **after** your analysis pipeline has produced:
- a `.md` file with the narrative text
- one or more `.png` chart images (from `stat_runtime`, matplotlib, or any other tool)

The tool reads these files, embeds charts inline as base64 data URIs, and writes
a single `.html` file that can be opened in any browser without any external assets.

## What it does NOT do

- Does not generate charts — chart generation is the responsibility of upstream tools.
- Does not run analysis or produce data — it is a pure assembler/renderer.

## Themes

| Theme | Style | Use case |
|-------|-------|----------|
| `executive` | Dark-blue accent, Inter font, rounded card | Board reports, strategy documents |
| `financial` | Teal accent, tabular numbers, left-border h2 | KPI dashboards, P&L reviews |
| `technical` | Dark mode, cyan accent, monospace headings | Engineering reports, incident reviews |
| `research` | Serif (Georgia), rust accent, academic spacing | White papers, market research |
