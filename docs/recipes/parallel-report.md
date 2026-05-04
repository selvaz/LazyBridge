# Parallel report pipeline

End-to-end recipe: three regional research agents run in parallel, each
contributing text + chart fragments via `fragment_tools(bus)`; a
synthesiser drafts the executive summary by reading what they wrote;
the final Step exports HTML + Reveal.js slides + PDF.

The shape mirrors `examples/parallel_report_pipeline.py` in the repo —
that file is runnable without LLM credentials (it uses Python-side
fragment appends).  The recipe below shows the **agent-driven** version
where the LLMs do the work.

## Goal

A daily-news-style report with a fixed outline:

```
1. Executive Summary
2. United States
3. China
4. India
5. Outlook
```

Three parallel "researcher" agents each fill their region.  A
"synthesiser" agent reads the assembled fragments and writes 1 + 5 (the
sections that depend on what others produced).

## Code

```python
from lazybridge import Agent, NativeTool, Plan, Step, from_parallel_all
from lazybridge.envelope import Envelope
from lazybridge.external_tools.report_builder import (
    FragmentBus,
    OutlineAssembler,
    fragment_tools,
)

OUTLINE = {
    "1.exec":     "Executive Summary",
    "2.us":       "United States",
    "3.cn":       "China",
    "4.in":       "India",
    "5.outlook":  "Outlook",
}

bus = FragmentBus("daily-news-2026-05-04", assembler=OutlineAssembler(OUTLINE))


def researcher(region_section: str, region_label: str, model: str) -> Agent:
    return Agent(
        name=f"research-{region_section}",
        model=model,
        system=(
            f"You research today's most important {region_label} news.\n"
            f"For each story:\n"
            f"  - Call append_text(heading=<headline>, body_markdown=<3-5 paragraphs>, "
            f"section='{region_section}', citations=<resolve via cite_url>).\n"
            f"  - When numbers are available, call append_chart with a Vega-Lite spec.\n"
            f"  - When a structured comparison helps, call append_table.\n"
            f"Use cite_url(<source URL>) before any quantitative claim and pass the "
            f"returned citation dict to append_text(citations=[...])."
        ),
        tools=[
            NativeTool.WEB_SEARCH,
            *fragment_tools(bus, default_section=region_section, step_name=region_section),
        ],
    )


synthesiser = Agent(
    name="synth",
    model="claude-sonnet-4-6",
    system=(
        "You write the executive summary and outlook for today's news report.\n"
        "1. Call list_fragments() to read what the regional agents wrote.\n"
        "2. Append the executive summary via append_text(section='1.exec', "
        "   body_markdown=<one tight paragraph + 3 bullets>).\n"
        "3. Append the outlook via append_text(section='5.outlook', "
        "   body_markdown=<one short paragraph>).\n"
        "Do not duplicate region content — synthesise, don't copy."
    ),
    tools=fragment_tools(bus, default_section="1.exec", step_name="synth"),
)


def export(env: Envelope) -> Envelope:
    paths = bus.export(
        ["html", "pdf", "revealjs"],
        "./out",
        title="Daily Global News — 2026-05-04",
        theme="cosmo",
        backend="auto",
    )
    return Envelope(payload=str({k: str(v) for k, v in paths.items()}))


plan = Plan(
    Step(researcher("2.us", "United States", "claude-haiku-4-5"), parallel=True, name="us"),
    Step(researcher("3.cn", "China",         "gemini-2.5-flash"), parallel=True, name="cn"),
    Step(researcher("4.in", "India",         "gpt-5-mini"),       parallel=True, name="in"),
    Step(synthesiser, task=from_parallel_all("us"), name="synth"),
    Step(export, name="export"),
)

agent = Agent(name="daily-news-pipeline", engine=plan)
agent("Today's daily news.")
```

## What's happening

* **Three parallel Steps.**  All three `parallel=True` siblings run in
  one band via `asyncio.gather`.  They share `bus` and append into it
  concurrently — the bus's CAS-backed append guarantees no losses.
* **Provider rotation.**  Each region uses a different provider so
  rate-limit budgets are independent.  This matters at scale; for
  prototyping any model works.
* **`from_parallel_all("us")`.**  The synthesiser's input task is the
  aggregated text from all three siblings.  We don't actually use that
  text — we pass the synthesiser the bus's `list_fragments()` result —
  but `from_parallel_all` is what makes the synthesiser run *after* the
  researchers, not in parallel with them.
* **Outline assembler.**  Sections 2/3/4 land where the agents tagged
  them; section 1 (exec) and 5 (outlook) come from the synthesiser.
  The outline mapping fixes the rendered order regardless of agent
  ordering.
* **Auto backend.**  When Quarto is installed, the export step uses it
  (proper Bootswatch theme, citeproc citations, Typst PDF, real
  Reveal.js).  When it isn't, `backend="auto"` falls back to WeasyPrint
  + pypandoc + a static reveal.js bundle.

## Add provenance + cost rollup

Every fragment is stamped with the `step_name` from `fragment_tools`.
For richer provenance (model, tokens, cost), have your agent code
copy from the run envelope:

```python
def my_research_step(env: Envelope) -> Envelope:
    out = research_agent(env.task)
    bus.append(Fragment(
        kind="text",
        body_md=out.text(),
        section="2.us",
        provenance=Provenance(
            step_name="us",
            agent_name="research_agent",
            model=out.metadata.model,
            tokens_in=out.metadata.input_tokens,
            tokens_out=out.metadata.output_tokens,
            cost_usd=out.metadata.cost_usd,
            latency_ms=out.metadata.latency_ms,
        ),
    ))
    return out
```

The assembler aggregates these into `AssembledReport.metadata` (totals)
and `AssembledReport.provenance_log` (per-fragment trail) — both render
in the audit table at the end of the report.

## Resume after a crash

If the process dies mid-run, point a fresh bus + Plan at the same
SQLite Store and resume:

```python
from lazybridge.store import Store

store = Store(db="./pipeline.sqlite")
bus = FragmentBus("daily-news-2026-05-04", store=store, assembler=OutlineAssembler(OUTLINE))

# Build the plan with checkpoint store + resume.
plan = Plan(
    ...,
    store=store,
    checkpoint_key="daily-news",
    resume=True,
)
```

Already-emitted fragments are persisted; replayed Steps that re-emit
the same `Fragment.id` no-op idempotently.  See [Plan with resume](plan-with-resume.md)
for the broader checkpoint pattern.

## Test it without LLM credentials

`examples/parallel_report_pipeline.py` shows the exact same pipeline
shape with Python-side appends instead of agent calls.  Run it locally:

```bash
python examples/parallel_report_pipeline.py --formats html,revealjs --backend weasyprint
ls parallel_reports/
# report.html  report.revealjs.html
```

The HTML carries:

* Outline-ordered sections (Executive Summary, US, CN, IN, Outlook).
* Vega-Lite chart of hyperscaler revenue (interactive).
* Pandoc-compatible callout box for the EU-tariff warning.
* Table of Indian IT services growth.
* `Sources` section with the bibliographic entries.
* `Audit Trail` section with per-fragment provenance.

## See also

- [FragmentBus](../guides/fragment-bus.md)
- [Assemblers](../guides/report-assemblers.md)
- [Decision: which report shape?](../decisions/report-shape.md)
- [Plan with resume](plan-with-resume.md)
