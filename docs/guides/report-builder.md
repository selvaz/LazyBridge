# Report builder

`lazybridge.external_tools.report_builder` ships **two complementary report-authoring
APIs** — pick the one that matches the shape of your pipeline.

| API                        | When to pick it                                      |
|----------------------------|------------------------------------------------------|
| `report_tools()`           | One agent assembles a complete report at the end of a pipeline. The LLM calls `generate_report` once with all sections / a Markdown file. Self-contained HTML (and optional PDF) with 4 themes. |
| `FragmentBus` + `fragment_tools()` | Multiple agents (often parallel Steps in a Plan) each contribute pieces — text, charts, tables, callouts — into a shared bus. An assembler recombines them; the result is rendered through Quarto (or a pure-Python fallback) into HTML, PDF, DOCX, or Reveal.js slides. Citations and per-fragment provenance are first-class. |

Both APIs co-exist; existing code using `report_tools()` keeps working
unchanged.

## Quick start — single-shot

```python
from lazybridge import Agent
from lazybridge.external_tools.report_builder import report_tools

agent = Agent("anthropic", tools=report_tools(output_dir="./out"))
agent(
    "Read analysis.md, embed the charts in ./out/charts/, "
    "save the polished report to ./out."
)
```

→ See [Single-shot generate_report](report-generate.md) for the full
field reference.

## Quick start — parallel fragments

```python
from lazybridge import Agent, Plan, Step
from lazybridge.external_tools.report_builder import (
    FragmentBus, fragment_tools, OutlineAssembler,
)

OUTLINE = {"1.exec": "Executive Summary", "2.body": "Findings", "3.outlook": "Outlook"}
bus = FragmentBus("audit", assembler=OutlineAssembler(OUTLINE))

researcher = Agent(
    model="claude-haiku-4-5",
    tools=fragment_tools(bus=bus, default_section="2.body", step_name="research"),
)

synth = Agent(
    model="claude-sonnet-4-6",
    tools=fragment_tools(bus=bus, default_section="1.exec", step_name="synth"),
)

plan = Plan(
    Step(researcher, parallel=True, name="research"),
    Step(synth, name="synth"),
    Step(lambda env: bus.export(["html", "pdf", "revealjs"], "./out", title="Audit"),
         name="export"),
)
Agent(name="audit", engine=plan)("Audit our customer churn last quarter.")
```

→ See [FragmentBus](fragment-bus.md) and [Parallel report recipe](../recipes/parallel-report.md).

## Why two APIs

The single-shot path is **the right answer** when one agent owns the
report end-to-end: a quarterly financial roll-up, an analyst's research
note, a saved-search summary.  It's the simplest tool surface and the
fastest path from prompt to PDF.

The fragment path is what you reach for when **the report is bigger
than any one agent**:

* Three regional research agents run in parallel; a synth agent writes
  the executive summary from what they collected.
* A nightly job that aggregates 50 SKU-level analyses into one document
  without holding all 50 contexts at once.
* A long-running pipeline that survives a crash and resumes — fragments
  already emitted are persisted in the `Store`.
* Anything where you want a **per-fragment audit trail** (which agent,
  which model, how many tokens, how long, citing what) showing in the
  rendered report.

## Architecture map

```
┌──────────────────────────── lazybridge.external_tools.report_builder ────────────────────────────┐
│                                                                                       │
│  ┌─ Single-shot path ────────────────────────────────────────────────────────────┐    │
│  │  report_tools(out_dir)  ─►  generate_report(...)  ─►  HTML / PDF              │    │
│  │                              ↑                                                 │    │
│  │                              │ uses                                            │    │
│  │                              ↓                                                 │    │
│  │                       renderer.py (Markdown → bleach → Jinja2 → Themes)        │    │
│  └────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                       │
│  ┌─ Fragment path ─────────────────────────────────────────────────────────────────┐  │
│  │                                                                                 │  │
│  │   ┌─────────────┐   ┌──────────────┐    ┌───────────────┐   ┌────────────┐      │  │
│  │   │  Step       │   │  Step        │    │  Step         │   │  Step      │      │  │
│  │   │ (parallel)  │   │  (parallel)  │ …  │ (synthesis)   │ → │ export()   │      │  │
│  │   │  └─ Agent   │   │   └─ Agent   │    │   └─ Agent    │   │            │      │  │
│  │   │     fragment_tools(bus)…                                              │      │  │
│  │   └─────┬───────┘   └─────┬────────┘    └───────┬───────┘   └──────┬─────┘      │  │
│  │         └─────────────────┴────────────────────┘                   │            │  │
│  │                                                                     ▼            │  │
│  │   ┌─ FragmentBus ──────────────────────────────────────────────────────┐         │  │
│  │   │  thread-safe append / read; backed by lazybridge.store.Store       │         │  │
│  │   └────────────────────────────────────────┬───────────────────────────┘         │  │
│  │                                            │ assembler.assemble()                │  │
│  │                                            ▼                                     │  │
│  │   ┌─ AssembledReport ──────────────────────────────────────────────────┐         │  │
│  │   │  sections (tree) + citations + provenance_log + metadata           │         │  │
│  │   └────────────────────────────────────────┬───────────────────────────┘         │  │
│  │                                            │ exporter.export()                   │  │
│  │                                            ▼                                     │  │
│  │   ┌─ Exporter ─────────────────────────────────────────────────────────┐         │  │
│  │   │  Quarto (primary)        WeasyPrint + pypandoc (fallback)          │         │  │
│  │   │   ├─ HTML (Bootswatch)    ├─ HTML (4 legacy themes)                │         │  │
│  │   │   ├─ PDF (Typst)          ├─ PDF (WeasyPrint paged-media)          │         │  │
│  │   │   ├─ DOCX (citeproc)      ├─ DOCX (pypandoc shell-out)             │         │  │
│  │   │   └─ Reveal.js slides     └─ Reveal.js (CDN one-file bundle)       │         │  │
│  │   └────────────────────────────────────────────────────────────────────┘         │  │
│  │                                                                                  │  │
│  └──────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## Public surface at a glance

```python
from lazybridge.external_tools.report_builder import (
    # Single-shot path (unchanged from 1.0)
    report_tools,
    ChartRef,
    ReportResult,

    # Fragment path — runtime
    FragmentBus,
    fragment_tools,

    # Fragment path — schema
    Fragment,
    Citation,
    Provenance,
    ChartSpec,
    TableSpec,

    # Fragment path — assembly
    Assembler,
    AssembledReport,
    BlackboardAssembler,
    OutlineAssembler,
    RenderedSection,
)
```

## Installation

```bash
pip install lazybridge[report]                 # core deps (markdown, bleach, jinja2)
pip install lazybridge[report-charts]          # vl-convert-python + plotly
pip install lazybridge[report-citations]       # habanero + citeproc-py + httpx
pip install lazybridge[report-fallback]        # weasyprint + pypandoc + python-docx
pip install lazybridge[pdf]                    # weasyprint only (legacy single-shot PDF)

# Quarto CLI (system, not pip-installable):
#   macOS:  brew install quarto
#   Linux:  download .deb / .rpm from https://quarto.org/docs/get-started/
#   Verify: quarto --version
```

The Quarto CLI is **only required for the primary fragment-path renderer**.
When it's missing, `bus.export(..., backend="auto")` falls back to the
WeasyPrint pure-Python path automatically.  `report_tools()` doesn't need
Quarto at all.

## Where to next

- [FragmentBus](fragment-bus.md) — thread-safety, persistence, idempotency
- [Fragment schema](report-fragments.md) — Fragment, Citation, Provenance, ChartSpec, TableSpec
- [Assemblers](report-assemblers.md) — Blackboard vs Outline
- [Chart contract](report-charts.md) — Vega-Lite + Plotly conventions
- [Citations](report-citations.md) — Crossref / OpenAlex enrichment, CSL-JSON
- [Exporters](report-exporters.md) — Quarto + WeasyPrint backends, format-by-format
- [Parallel report recipe](../recipes/parallel-report.md) — full end-to-end example
- [Decision: which report shape?](../decisions/report-shape.md)
