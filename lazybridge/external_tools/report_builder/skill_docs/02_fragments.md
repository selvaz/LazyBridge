# Fragment-based parallel reports

Use this when **multiple Steps in a Plan each contribute pieces of a single
final report** — research agents run in parallel, each writing prose,
charts, tables, callouts, and citations into a shared bus, and a final
Step renders the whole thing in HTML / PDF / DOCX / Reveal.js.

## The pieces

| Type             | Role                                              |
|------------------|---------------------------------------------------|
| `FragmentBus`    | Thread-safe collector. One per report.            |
| `Fragment`       | Typed payload: text / chart / table / callout.    |
| `fragment_tools` | LLM-callable tools that append fragments.         |
| `Assembler`      | Reduces fragments to a structured report tree.    |
| `Exporter`       | Renders the tree to HTML / PDF / DOCX / Reveal.   |

## Minimal example

```python
from lazybridge import Plan, Step, Agent
from lazybridge.external_tools.report_builder import (
    FragmentBus, fragment_tools, OutlineAssembler,
)

# 1. Declare an outline so every Step knows where its contributions land.
OUTLINE = {
    "1.exec":   "Executive Summary",
    "2.us":     "United States",
    "2.cn":     "China",
    "3.outlook": "Outlook",
}
bus = FragmentBus("daily-news", assembler=OutlineAssembler(OUTLINE))

# 2. Each Step gets a fragment_tools list scoped to its section.
def researcher(region_section: str) -> Agent:
    return Agent(
        model="anthropic:claude-haiku-4-5",
        system=(
            f"Research today's news for section {region_section}.  "
            f"Call append_text and append_chart with section='{region_section}'."
        ),
        tools=fragment_tools(bus, default_section=region_section),
    )

# 3. Final Step exports.  Quarto first, WeasyPrint fallback.
def export_step(_env):
    return bus.export(
        ["html", "pdf", "revealjs"],
        "./out",
        title="Daily Global News",
        theme="cosmo",
    )

plan = Plan(
    Step(researcher("2.us"), parallel=True, name="us"),
    Step(researcher("2.cn"), parallel=True, name="cn"),
    Step(export_step),
)
plan.run("Today's news.")
```

## Picking an assembler

* `BlackboardAssembler()` — emergent pipelines.  Groups by `section`
  string alphabetically.  Use when you don't know the report shape upfront
  (news digests, audit summaries, scratch reports).
* `OutlineAssembler({sid: heading, ...})` — STORM-shaped pipelines.  Pass
  a dotted-path outline and have agents tag fragments with matching ids.
  Use when the report has a fixed shape (executive summaries, research
  reports, financial briefings).

## Citations + bibliography

Each fragment carries a `citations: list[Citation]`.  Use the `cite_url`
tool to enrich a URL or DOI through Crossref / OpenAlex (requires
`pip install lazybridge[report-citations]`).  At export time the Quarto
exporter writes a CSL-JSON file and Pandoc citeproc resolves any
`[@citation_key]` markers in your prose into formatted footnotes plus a
Sources section.  The WeasyPrint fallback renders citations as a plain
list at the end of the document.

## Provenance / audit trail

Every fragment is stamped with a `Provenance` (step name, timestamp).  The
assembler aggregates these into a per-fragment audit table at the end of
the report — one row per fragment, with model / tokens / cost / latency
columns when supplied.  Useful for compliance and for debugging
multi-agent pipelines.

## Backends

* **Quarto** is the primary path: HTML + PDF (Typst) + DOCX + Reveal.js
  from one source via Pandoc citeproc.  Install the CLI from
  <https://quarto.org/docs/get-started/>.  We auto-detect it on `$PATH`.
* **WeasyPrint** is the pure-Python fallback for `backend="weasyprint"` or
  when Quarto isn't installed.  Slightly weaker output (no Bootswatch
  themes, no Mermaid auto-rendering, DOCX requires a separate `pandoc`
  binary), but `pip install lazybridge[report-fallback]` and you're done.

## Persistence

The bus is backed by `lazybridge.store.Store`.  Pass an existing Store to
share state with a Plan's checkpointing — fragments survive a crash and
resume picks them up.  Default is in-memory.
