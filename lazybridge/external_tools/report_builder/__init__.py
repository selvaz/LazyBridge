"""report_builder — fragment-based parallel report assembly + classic single-shot tool.

Two complementary entry points ship in this extension:

1. **Single-shot tool** (the original ``report_tools(output_dir=)`` flow).  An LLM
   hands a fully-formed report — sections or a Markdown file plus chart
   PNGs — to ``generate_report`` once at the end of a pipeline.  Best when
   one agent does the work end-to-end.

2. **Parallel-fragment workflow** (the new ``FragmentBus`` + ``fragment_tools(bus=)``).
   Each Step in a Plan emits typed fragments (text / chart / table /
   callout) into a shared bus.  At the end an ``Assembler``
   (``BlackboardAssembler`` or ``OutlineAssembler``) recombines them; an
   ``Exporter`` writes HTML, PDF, DOCX, and Reveal.js slides via Quarto
   with a pure-Python (WeasyPrint + Pandoc) fallback.

Quick start (parallel-fragment workflow)::

    from lazybridge import Plan, Step, Agent
    from lazybridge.external_tools.report_builder import (
        FragmentBus, fragment_tools, OutlineAssembler,
    )

    bus = FragmentBus("daily-news", assembler=OutlineAssembler({
        "1.exec": "Executive Summary",
        "2.us":   "United States",
    }))

    researcher = Agent(
        model="anthropic:claude-haiku-4-5",
        tools=fragment_tools(bus=bus, default_section="2.us", step_name="us"),
    )

    plan = Plan(
        Step(researcher, parallel=True, name="us"),
        Step(lambda env: bus.export(["html","pdf","revealjs"], "./out", title="News")),
    )
    plan.run("Today's news.")

Install extras (additive over ``[report]``)::

    pip install 'lazybridge[report,report-charts,report-citations,report-fallback]'
    # And install the Quarto CLI separately for the primary render path:
    # macOS: brew install quarto / Linux: download .deb from quarto.org
"""

from lazybridge.external_tools.report_builder.assemblers import (
    AssembledReport,
    Assembler,
    BlackboardAssembler,
    OutlineAssembler,
    RenderedSection,
)
from lazybridge.external_tools.report_builder.bus import FragmentBus
from lazybridge.external_tools.report_builder.fragments import (
    ChartSpec,
    Citation,
    Fragment,
    Provenance,
    TableSpec,
)
from lazybridge.external_tools.report_builder.schemas import ChartRef, ReportResult
from lazybridge.external_tools.report_builder.tools import fragment_tools, report_tools

__all__ = [
    # Original single-shot API
    "report_tools",
    "ChartRef",
    "ReportResult",
    # Fragment workflow — runtime
    "FragmentBus",
    "fragment_tools",
    # Fragment workflow — schema
    "Fragment",
    "Citation",
    "Provenance",
    "ChartSpec",
    "TableSpec",
    # Fragment workflow — assembly
    "Assembler",
    "AssembledReport",
    "BlackboardAssembler",
    "OutlineAssembler",
    "RenderedSection",
]
