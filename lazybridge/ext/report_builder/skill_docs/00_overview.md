# report_builder — Overview

Two complementary entry points ship in this extension.  Pick whichever
matches the shape of your pipeline.

## 1. Single-shot tool — `report_tools()` / `generate_report`

Original API.  An LLM hands a fully-formed report — sections or a
Markdown file plus chart PNGs — to `generate_report` once at the end of a
pipeline.  Best when one agent does the work end-to-end.

See [`01_usage.md`](01_usage.md).

## 2. Parallel-fragment workflow — `FragmentBus` + `fragment_tools()`

New.  Each Step in a Plan emits typed fragments (text / chart / table /
callout) into a shared bus; an `Assembler` recombines them; an
`Exporter` writes HTML, PDF, DOCX, and Reveal.js slides via the Quarto
CLI (Pandoc citeproc, Bootswatch themes) with a pure-Python fallback.

Use this when:

* multiple agents contribute pieces of the same report in parallel,
* you want auto-resolved citations + a per-fragment audit trail,
* you need interactive Vega-Lite / Plotly charts (not pre-baked PNGs),
* you want one source rendering to four formats.

See [`02_fragments.md`](02_fragments.md) and
[`03_charts.md`](03_charts.md).

## What the extension still does NOT do

* Run analysis or fetch data — the bus collects the work other Steps do.
* Execute LLM-emitted Python code for charts.  Charts are emitted as
  JSON specs (Vega-Lite or Plotly), validated, and rendered server-side.

## Themes

The single-shot tool ships four CSS themes (`executive`, `financial`,
`technical`, `research`).  The fragment workflow goes through Quarto and
exposes the full Bootswatch theme catalog (`cosmo`, `flatly`, `litera`,
`darkly`, `lux`, …).  When Quarto isn't available, the WeasyPrint fallback
maps Bootswatch names onto the four legacy themes.
