"""Parallel-fragment report pipeline — runnable demo.

Demonstrates the new fragment-based reporting workflow without requiring an
LLM API key.  Three "research" Steps run in parallel, each appending text +
chart + table fragments into a shared :class:`FragmentBus`.  A final step
exports the assembled report to HTML (and optionally PDF, DOCX, Reveal.js
when the relevant deps are installed).

Run::

    python examples/parallel_report_pipeline.py
    python examples/parallel_report_pipeline.py --formats html,pdf
    python examples/parallel_report_pipeline.py --backend weasyprint

Compare the structure with ``examples/daily_news_report.py`` (734 lines):
this file does the same fragment-assembly pattern in well under 200 lines
because the framework now owns the bus / assembler / exporter machinery.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from lazybridge import Agent, Plan, Step
from lazybridge.envelope import Envelope
from lazybridge.ext.report_builder import (
    Citation,
    Fragment,
    FragmentBus,
    OutlineAssembler,
    Provenance,
)
from lazybridge.ext.report_builder.fragments import ChartSpec, TableSpec


OUTLINE = {
    "1.exec": "Executive Summary",
    "2.us": "United States",
    "2.cn": "China",
    "2.in": "India",
    "3.outlook": "Outlook",
}


def _us_research(env: Envelope, *, bus: FragmentBus) -> Envelope:
    """Stand-in for an LLM agent contributing US fragments."""
    bus.append(
        Fragment(
            kind="text",
            heading="Tech earnings",
            section="2.us",
            body_md=(
                "Mega-cap tech reported a strong quarter, beating consensus "
                "on AI-related revenue [@bigtech2026].  Cloud margins remain "
                "the dominant driver."
            ),
            citations=[
                Citation(
                    key="bigtech2026",
                    title="Mega-cap tech Q1 2026 earnings",
                    url="https://example.com/bigtech-q1",
                    year=2026,
                )
            ],
            provenance=Provenance(step_name="us", agent_name="researcher", model="claude-haiku-4-5"),
        )
    )
    bus.append(
        Fragment(
            kind="chart",
            heading="Hyperscaler revenue (USD bn)",
            section="2.us",
            chart=ChartSpec(
                engine="vega-lite",
                title="Hyperscaler revenue by quarter",
                spec={
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "quarter", "type": "ordinal"},
                        "y": {"field": "revenue_usd_bn", "type": "quantitative"},
                        "color": {"field": "company", "type": "nominal"},
                    },
                },
                data=[
                    {"quarter": "Q4'25", "revenue_usd_bn": 27.0, "company": "AWS"},
                    {"quarter": "Q4'25", "revenue_usd_bn": 35.0, "company": "Azure"},
                    {"quarter": "Q4'25", "revenue_usd_bn": 12.0, "company": "GCP"},
                    {"quarter": "Q1'26", "revenue_usd_bn": 28.5, "company": "AWS"},
                    {"quarter": "Q1'26", "revenue_usd_bn": 38.0, "company": "Azure"},
                    {"quarter": "Q1'26", "revenue_usd_bn": 13.5, "company": "GCP"},
                ],
            ),
            provenance=Provenance(step_name="us", agent_name="researcher", model="claude-haiku-4-5"),
        )
    )
    return Envelope(payload="us-done")


def _cn_research(env: Envelope, *, bus: FragmentBus) -> Envelope:
    bus.append(
        Fragment(
            kind="text",
            heading="EV exports",
            section="2.cn",
            body_md=(
                "China's electric-vehicle exports continued to climb, driven "
                "by competitive pricing in EU and ASEAN markets."
            ),
            provenance=Provenance(step_name="cn", agent_name="researcher", model="gemini-2.5-flash"),
        )
    )
    bus.append(
        Fragment(
            kind="callout",
            section="2.cn",
            callout_style="warning",
            body_md=(
                "EU anti-subsidy tariffs introduced in late 2025 are starting "
                "to bite — Q1'26 export volumes are flat in EU but up sharply "
                "elsewhere."
            ),
            provenance=Provenance(step_name="cn", agent_name="researcher", model="gemini-2.5-flash"),
        )
    )
    return Envelope(payload="cn-done")


def _in_research(env: Envelope, *, bus: FragmentBus) -> Envelope:
    bus.append(
        Fragment(
            kind="text",
            heading="Services exports",
            section="2.in",
            body_md=(
                "India's IT services sector reported steady AI-driven demand, "
                "with mid-cap names outperforming the top tier on margin."
            ),
            provenance=Provenance(step_name="in", agent_name="researcher", model="gpt-5-mini"),
        )
    )
    bus.append(
        Fragment(
            kind="table",
            heading="Top-line growth",
            section="2.in",
            table=TableSpec(
                headers=["Company", "YoY Revenue", "Op. Margin"],
                rows=[
                    ["TCS", "+11%", "24.1%"],
                    ["Infosys", "+9%", "21.3%"],
                    ["Persistent Systems", "+22%", "16.0%"],
                ],
                caption="Q1'26 reported figures (USD-equivalent)",
            ),
            provenance=Provenance(step_name="in", agent_name="researcher", model="gpt-5-mini"),
        )
    )
    return Envelope(payload="in-done")


def _exec_summary(env: Envelope, *, bus: FragmentBus) -> Envelope:
    """Read what the researchers wrote, draft an executive summary."""
    headings = [f.heading for f in bus.fragments() if f.heading]
    bus.append(
        Fragment(
            kind="text",
            section="1.exec",
            order_hint=0.0,
            body_md=(
                "Cross-region demand for AI infrastructure was the dominant "
                "theme this quarter.  US hyperscalers accelerated; China's "
                "industrial base extended its export lead in EVs despite "
                "tariff headwinds; India's services sector benefited from "
                "AI implementation work.  Coverage included: "
                + ", ".join(headings or ["—"])
                + "."
            ),
            provenance=Provenance(step_name="exec", agent_name="synthesiser", model="claude-sonnet-4-6"),
        )
    )
    bus.append(
        Fragment(
            kind="text",
            section="3.outlook",
            body_md=(
                "Capex for AI training compute remains the swing factor.  "
                "Watch for guidance updates in Q2'26 earnings season."
            ),
            provenance=Provenance(step_name="outlook", agent_name="synthesiser", model="claude-sonnet-4-6"),
        )
    )
    return Envelope(payload="exec-done")


def _export(env: Envelope, *, bus: FragmentBus, out_dir: Path, formats: list[str], backend: str) -> Envelope:
    produced = bus.export(
        formats,  # type: ignore[arg-type]
        out_dir,
        title="Daily Global Briefing — {}".format(datetime.now(timezone.utc).date().isoformat()),
        theme="cosmo",
        backend=backend,  # type: ignore[arg-type]
    )
    print("Produced:")
    for fmt, path in produced.items():
        print(f"  {fmt:<10} {path}")
    return Envelope(payload=str({k: str(v) for k, v in produced.items()}))


def build_pipeline(bus: FragmentBus, out_dir: Path, formats: list[str], backend: str) -> Agent:
    """Wrap a Plan in an Agent so it has the standard sync ``__call__`` entrypoint.

    LazyBridge runs Plans through the Agent harness — this gives us the
    sync wrapper, the Session, the engine plumbing, etc. for free.
    """
    plan = Plan(
        Step(lambda env: _us_research(env, bus=bus), name="us", parallel=True),
        Step(lambda env: _cn_research(env, bus=bus), name="cn", parallel=True),
        Step(lambda env: _in_research(env, bus=bus), name="in", parallel=True),
        Step(lambda env: _exec_summary(env, bus=bus), name="exec"),
        Step(
            lambda env: _export(env, bus=bus, out_dir=out_dir, formats=formats, backend=backend),
            name="export",
        ),
    )
    return Agent(name="parallel-report-pipeline", engine=plan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel-fragment report pipeline demo.")
    parser.add_argument("--out", type=Path, default=Path("./parallel_reports"))
    parser.add_argument(
        "--formats",
        type=str,
        default="html,revealjs",
        help="Comma-separated subset of: html, pdf, docx, revealjs.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "quarto", "weasyprint"],
        default="auto",
        help="Render backend.  'auto' picks Quarto when on PATH, WeasyPrint otherwise.",
    )
    args = parser.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    bus = FragmentBus("parallel-demo", assembler=OutlineAssembler(OUTLINE))
    pipeline = build_pipeline(bus, args.out, formats, args.backend)
    pipeline("Today's briefing.")


if __name__ == "__main__":
    main()
