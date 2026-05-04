"""report_builder — professional HTML report assembler (domain example).

Domain example shipped with LazyBridge — not part of the framework contract.

Assembles self-contained HTML reports from LLM-authored Markdown files and
pre-generated chart PNG images produced by upstream tools (e.g. stat_runtime).
Charts are embedded as base64 data URIs and auto-placed inline next to the
best-matching section heading — no external files are needed to view the output.

Intended workflow::

    # 1. Analysis tools produce a Markdown narrative and chart PNGs
    # 2. generate_report assembles them into a polished HTML document

    from lazybridge import Agent
    from lazybridge.ext.report_builder import report_tools

    agent = Agent("anthropic", tools=report_tools("./reports"))
    agent(
        "Analyse the data, write a narrative in ./output/analysis.md, "
        "save chart PNGs to ./output/charts/, then call generate_report."
    )

Install::

    pip install lazybridge[report]

Available themes: executive, financial, technical, research.
"""

__stability__ = "domain"
__lazybridge_min__ = "1.0.0"

from lazybridge.ext.report_builder.schemas import ChartRef, ReportResult
from lazybridge.ext.report_builder.tools import report_tools

__all__ = ["report_tools", "ChartRef", "ReportResult"]
