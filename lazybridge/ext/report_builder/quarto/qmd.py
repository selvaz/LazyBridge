"""Render :class:`Fragment` and :class:`AssembledReport` to Quarto Markdown (.qmd).

Quarto-flavored Markdown adds:

* Pandoc citation syntax: ``[@key]``, ``[@key, p. 5]``.
* Fenced div callouts: ``::: {.callout-note} … :::``.
* Embedded Vega-Lite blocks via ``{vegalite}`` filter (rasterized for PDF
  by Quarto automatically).
* Embedded HTML for Plotly via raw ``{=html}`` blocks.

We emit these primitives directly from the typed Fragment/Section tree —
no LLM in the loop on the rendering path.
"""

from __future__ import annotations

import json
from typing import Iterable

from lazybridge.ext.report_builder.assemblers import AssembledReport, RenderedSection
from lazybridge.ext.report_builder.fragments import Fragment


# ---------------------------------------------------------------------------
# Fragment-level rendering
# ---------------------------------------------------------------------------


def _table_to_pipe(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    """Format a table as a Pandoc pipe table.

    Pipe tables don't allow line breaks within cells; we replace any
    embedded newlines with a single space so the output stays valid even
    when an LLM emits multi-line cells.
    """

    def _norm(cell: str) -> str:
        return str(cell).replace("\n", " ").replace("|", "\\|")

    head = "| " + " | ".join(_norm(h) for h in headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body_lines = ["| " + " | ".join(_norm(c) for c in row) + " |" for row in rows]
    table = "\n".join([head, sep, *body_lines])
    if caption:
        table = f"{table}\n\n: {caption}"
    return table


def render_fragment_to_qmd(fragment: Fragment) -> str:
    """Return a self-contained Markdown snippet for one fragment.

    The caller is responsible for placing this snippet under the right
    section heading; this function emits content + (optional) sub-heading
    only.  ``heading`` on a fragment becomes an h3 (sub-section); the
    section-level h2 is emitted by :func:`render_report_to_qmd`.
    """
    parts: list[str] = []
    if fragment.heading:
        parts.append(f"### {fragment.heading}")
        parts.append("")

    if fragment.kind == "text":
        parts.append(fragment.body_md or "")
    elif fragment.kind == "callout":
        style = fragment.callout_style or "note"
        body = (fragment.body_md or "").strip()
        # Quarto callouts: `::: {.callout-note}` … `:::`
        parts.append(f"::: {{.callout-{style}}}")
        parts.append(body)
        parts.append(":::")
    elif fragment.kind == "table":
        if fragment.table is None:
            parts.append("<!-- empty table fragment -->")
        else:
            parts.append(
                _table_to_pipe(fragment.table.headers, fragment.table.rows, fragment.table.caption)
            )
    elif fragment.kind == "chart":
        if fragment.chart is None:
            parts.append("<!-- empty chart fragment -->")
        else:
            parts.append(_render_chart_block(fragment))
    else:
        parts.append(f"<!-- unknown fragment kind: {fragment.kind} -->")

    parts.append("")  # trailing blank line — Pandoc cares about block separation
    return "\n".join(parts)


def _render_chart_block(fragment: Fragment) -> str:
    """Produce a Quarto/Pandoc snippet for a chart fragment.

    For Vega-Lite we use Quarto's native ``{vegalite}`` block — Quarto
    rasterises it for PDF/DOCX automatically.  For Plotly we emit a raw
    ``{=html}`` block carrying the interactive embed; this renders
    interactively in HTML and Reveal.js, and Pandoc strips it out for PDF/
    DOCX (the exporter pre-rasterises Plotly figures separately for those
    formats — handled by the exporter's chart preprocessor).
    """
    chart = fragment.chart
    assert chart is not None
    title = chart.title or fragment.heading or ""
    if chart.engine == "vega-lite":
        spec = dict(chart.spec)
        if chart.data is not None:
            spec["data"] = {"values": chart.data}
        spec_json = json.dumps(spec, default=str, indent=2)
        block = ["```{vegalite}", spec_json, "```"]
        if title:
            block.append("")
            block.append(f": {title}")
        return "\n".join(block)
    if chart.engine == "plotly":
        from lazybridge.ext.report_builder.charts import render_chart

        rendered = render_chart(chart)
        # Wrap the interactive embed in a raw HTML block so Pandoc preserves
        # it for HTML/Reveal output and ignores it for non-HTML targets.
        return "```{=html}\n" + rendered.html + "\n```"
    return f"<!-- unsupported chart engine: {chart.engine} -->"


# ---------------------------------------------------------------------------
# Report-level rendering
# ---------------------------------------------------------------------------


def _render_section(section: RenderedSection, depth: int = 2) -> str:
    """Render a section + its children + its fragments as Markdown."""
    parts: list[str] = []
    if section.heading:
        marker = "#" * min(max(depth, 1), 6)
        parts.append(f"{marker} {section.heading}")
        parts.append("")

    for fragment in section.fragments:
        parts.append(render_fragment_to_qmd(fragment))

    for child in section.children:
        parts.append(_render_section(child, depth=depth + 1))

    return "\n".join(parts)


def _render_provenance_appendix(report: AssembledReport) -> str:
    """Render the per-fragment provenance audit trail as a Pandoc table."""
    if not report.provenance_log:
        return ""
    headers = ["#", "Step", "Agent", "Model", "Tokens (in/out)", "Cost USD", "Latency (ms)"]
    rows: list[list[str]] = []
    for i, p in enumerate(report.provenance_log, start=1):
        tokens = (
            f"{p.tokens_in or 0}/{p.tokens_out or 0}"
            if (p.tokens_in is not None or p.tokens_out is not None)
            else "—"
        )
        rows.append(
            [
                str(i),
                p.step_name or "—",
                p.agent_name or "—",
                p.model or "—",
                tokens,
                f"{p.cost_usd:.4f}" if p.cost_usd is not None else "—",
                f"{p.latency_ms:.0f}" if p.latency_ms is not None else "—",
            ]
        )
    table = _table_to_pipe(headers, rows, caption="Per-fragment audit trail")
    return "## Audit Trail\n\n" + table + "\n"


def render_report_to_qmd(
    report: AssembledReport,
    *,
    bibliography_path: str | None = None,
    csl_style: str | None = None,
    extra_yaml: dict | None = None,
) -> str:
    """Render an :class:`AssembledReport` into a complete .qmd document.

    The output begins with a YAML front-matter block (Quarto consumes it),
    then walks the section tree, then appends the audit-trail appendix and
    the bibliography stub (Pandoc citeproc auto-fills the Sources section
    when ``bibliography:`` is set).
    """
    yaml_lines = ["---", f'title: "{_yaml_escape(report.title)}"']
    if report.metadata.get("author"):
        yaml_lines.append(f'author: "{_yaml_escape(report.metadata["author"])}"')
    if bibliography_path:
        yaml_lines.append(f"bibliography: {bibliography_path}")
    if csl_style:
        yaml_lines.append(f"csl: {csl_style}")
    if extra_yaml:
        for k, v in extra_yaml.items():
            yaml_lines.append(f"{k}: {v}")
    yaml_lines.append("---")
    yaml_lines.append("")

    body_parts: list[str] = list(yaml_lines)
    for section in report.sections:
        body_parts.append(_render_section(section, depth=2))

    audit = _render_provenance_appendix(report)
    if audit:
        body_parts.append(audit)

    if report.citations:
        # Pandoc citeproc auto-renders the bibliography under the heading
        # named in ``link-citations`` / ``reference-section-title``; we
        # provide a clear h2 anchor so the assembled report has it even if
        # the user opens the .qmd in another tool.
        body_parts.append("## Sources\n\n::: {#refs}\n:::\n")

    return "\n".join(body_parts).rstrip() + "\n"


def _yaml_escape(s: str) -> str:
    """Escape a string for inclusion inside double-quoted YAML."""
    return s.replace('"', '\\"')


def collect_chart_inputs(fragments: Iterable[Fragment]) -> list[Fragment]:
    """Return chart fragments — convenience for exporters that need pre-raster passes."""
    return [f for f in fragments if f.kind == "chart"]
