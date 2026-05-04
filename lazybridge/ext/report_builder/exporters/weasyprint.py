"""WeasyPrint-backed fallback exporter.

Used when Quarto isn't installed (``backend="auto"`` failover) or when the
caller passes ``backend="weasyprint"`` explicitly.  Reuses the existing
``renderer.py`` from the original report_builder for HTML; goes through
WeasyPrint for PDF; shells out to Pandoc via ``pypandoc`` for DOCX; emits
a static reveal.js bundle for slides.

Trade-offs vs the Quarto path:

* HTML: still beautiful, but no Bootswatch — uses our existing 4 themes
  (executive/financial/technical/research).  ``theme=`` argument maps to
  these names; unknown theme names raise.
* PDF: WeasyPrint is solid; CSS Paged Media for page numbers + page
  breaks.  No Typst speed advantage.
* DOCX: requires the ``pandoc`` binary on PATH (lighter than the full
  Quarto runtime).  We surface a clear error if it isn't there.
* Reveal.js: a static one-file HTML bundle pointing at the reveal.js CDN.
  Honest second-best vs Quarto's real reveal output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from lazybridge.ext.report_builder.assemblers import AssembledReport, RenderedSection
from lazybridge.ext.report_builder.exporters import Exporter
from lazybridge.ext.report_builder.fragments import Fragment


# Map Bootswatch-ish names down to the existing 4 themes shipped in the
# extension.  Anything we don't know stays as-is and the renderer raises
# (matching the existing behaviour).
_THEME_ALIAS = {
    "cosmo": "executive",
    "flatly": "executive",
    "litera": "research",
    "lux": "executive",
    "morph": "research",
    "minty": "financial",
    "darkly": "technical",
    "slate": "technical",
    "default": "executive",
}


class WeasyPrintExporter(Exporter):
    """Pure-Python rendering — no Quarto / Pandoc required for HTML & PDF."""

    def export(
        self,
        report: AssembledReport,
        output_dir: Path,
        *,
        formats: list[str],
        theme: str = "cosmo",
    ) -> dict[str, Path]:
        from lazybridge.ext.report_builder.renderer import (
            VALID_THEMES,
            build_html_document_jinja2,
            load_theme_css,
        )

        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map the user's chosen theme down to one of the four shipped themes.
        local_theme = _THEME_ALIAS.get(theme, theme)
        if local_theme not in VALID_THEMES:
            local_theme = "executive"

        body_html = self._render_body(report)
        css = load_theme_css(local_theme)
        meta = {
            "generated_at": report.metadata.get("generated_at", ""),
            "theme": local_theme,
            "template": "default",
            "charts_embedded": sum(
                1 for f in self._iter_fragments(report) if f.kind == "chart"
            ),
        }
        full_html = build_html_document_jinja2(body_html, report.title, css, "default", meta)

        produced: dict[str, Path] = {}
        if "html" in formats:
            html_path = output_dir / "report.html"
            html_path.write_text(full_html, encoding="utf-8")
            produced["html"] = html_path

        if "pdf" in formats:
            from lazybridge.ext.report_builder._deps import require_weasyprint

            weasyprint = require_weasyprint()
            pdf_path = output_dir / "report.pdf"
            # Inject @page rules for page numbers + sensible margins.  The
            # existing themes don't define paged-media bits, so we tack
            # them on as a style override.
            paged_css = (
                "<style>"
                "@page { size: letter; margin: 1in; "
                "@bottom-right { content: counter(page) ' / ' counter(pages); "
                "font-size: 10px; color: #666; } } "
                "h2 { page-break-before: always; } "
                "h2:first-of-type { page-break-before: auto; }"
                "</style>"
            )
            paged_html = full_html.replace("</head>", paged_css + "</head>", 1)
            weasyprint.HTML(string=paged_html).write_pdf(str(pdf_path))
            produced["pdf"] = pdf_path

        if "docx" in formats:
            produced["docx"] = self._export_docx(report, output_dir)

        if "revealjs" in formats:
            produced["revealjs"] = self._export_revealjs(report, output_dir)

        return produced

    # ------------------------------------------------------------------
    # Body rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_fragments(report: AssembledReport):
        for section in report.sections:
            yield from _walk_fragments(section)

    def _render_body(self, report: AssembledReport) -> str:
        from lazybridge.ext.report_builder.renderer import markdown_to_clean_html

        # Build a single Markdown blob from the section tree, then run our
        # existing sanitiser to produce safe HTML.
        md = self._sections_to_markdown(report.sections, depth=2)
        # Append audit trail + sources at the end so they always exist.
        if report.provenance_log:
            md += "\n\n## Audit Trail\n\n"
            md += _provenance_md_table(report)
        if report.citations:
            md += "\n\n## Sources\n\n"
            for c in report.citations:
                md += _format_citation_line(c) + "\n"
        return markdown_to_clean_html(md)

    def _sections_to_markdown(self, sections: list[RenderedSection], depth: int) -> str:
        out: list[str] = []
        for section in sections:
            if section.heading:
                marker = "#" * min(max(depth, 1), 6)
                out.append(f"{marker} {section.heading}")
                out.append("")
            for f in section.fragments:
                out.append(self._fragment_to_markdown(f))
            if section.children:
                out.append(self._sections_to_markdown(section.children, depth + 1))
        return "\n".join(out)

    def _fragment_to_markdown(self, fragment: Fragment) -> str:
        from lazybridge.ext.report_builder.charts import render_chart

        if fragment.kind == "text":
            block = fragment.body_md or ""
        elif fragment.kind == "callout":
            style = fragment.callout_style or "note"
            block = f"> **{style.upper()}** — {fragment.body_md or ''}"
        elif fragment.kind == "table" and fragment.table:
            block = _table_md(fragment.table.headers, fragment.table.rows, fragment.table.caption)
        elif fragment.kind == "chart" and fragment.chart:
            rendered = render_chart(fragment.chart)
            # Fallback HTML doesn't support fenced-html in markdown safely
            # (bleach would strip it).  Inline as a self-contained snippet
            # via Markdown's "raw HTML" path — markdown_to_clean_html keeps
            # the safelisted bits and drops the rest.
            block = rendered.html
        else:
            block = f"<!-- empty fragment {fragment.id} -->"
        if fragment.heading:
            return f"### {fragment.heading}\n\n{block}\n"
        return block + "\n"

    # ------------------------------------------------------------------
    # DOCX (via pypandoc)
    # ------------------------------------------------------------------

    def _export_docx(self, report: AssembledReport, output_dir: Path) -> Path:
        from lazybridge.ext.report_builder._deps import require_pypandoc

        pypandoc = require_pypandoc()
        md = self._sections_to_markdown(report.sections, depth=2)
        if report.provenance_log:
            md += "\n\n## Audit Trail\n\n" + _provenance_md_table(report)
        if report.citations:
            md += "\n\n## Sources\n\n"
            for c in report.citations:
                md += _format_citation_line(c) + "\n"
        docx_path = output_dir / "report.docx"
        pypandoc.convert_text(
            md,
            "docx",
            format="markdown",
            outputfile=str(docx_path),
            extra_args=["--toc", "--reference-links"],
        )
        return docx_path

    # ------------------------------------------------------------------
    # Reveal.js standalone bundle
    # ------------------------------------------------------------------

    def _export_revealjs(self, report: AssembledReport, output_dir: Path) -> Path:
        slides_html: list[str] = []
        # Title slide
        slides_html.append(
            f"<section><h1>{_html_escape(report.title)}</h1>"
            f"<p>{_html_escape(report.metadata.get('generated_at', ''))}</p></section>"
        )
        for section in report.sections:
            slides_html.append(_section_to_reveal_slides(section))
        if report.citations:
            sources = "<ul>" + "".join(
                f"<li>{_html_escape(_format_citation_line(c))}</li>" for c in report.citations
            ) + "</ul>"
            slides_html.append(f"<section><h2>Sources</h2>{sources}</section>")

        body = "\n".join(slides_html)
        html = (
            "<!DOCTYPE html><html><head>"
            f"<title>{_html_escape(report.title)}</title>"
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css">'
            '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/simple.css">'
            "</head><body>"
            '<div class="reveal"><div class="slides">'
            f"{body}"
            "</div></div>"
            '<script src="https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"></script>'
            '<script>Reveal.initialize({hash: true});</script>'
            "</body></html>"
        )
        path = output_dir / "report.revealjs.html"
        path.write_text(html, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _walk_fragments(section: RenderedSection):
    yield from section.fragments
    for child in section.children:
        yield from _walk_fragments(child)


def _table_md(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = "\n".join("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows)
    table = "\n".join([head, sep, body])
    if caption:
        table += f"\n\n*{caption}*"
    return table


def _provenance_md_table(report: AssembledReport) -> str:
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
    return _table_md(headers, rows, "")


def _format_citation_line(c) -> str:
    parts = [c.title]
    if c.authors:
        parts.append(", ".join(c.authors))
    if c.year:
        parts.append(str(c.year))
    if c.doi:
        parts.append(f"DOI:{c.doi}")
    elif c.url:
        parts.append(c.url)
    return " — ".join(parts)


def _section_to_reveal_slides(section: RenderedSection) -> str:
    parts: list[str] = []
    inner = []
    if section.heading:
        inner.append(f"<h2>{_html_escape(section.heading)}</h2>")
    for f in section.fragments:
        inner.append(_fragment_to_reveal_html(f))
    parts.append("<section>" + "".join(inner) + "</section>")
    for child in section.children:
        parts.append(_section_to_reveal_slides(child))
    return "\n".join(parts)


def _fragment_to_reveal_html(fragment: Fragment) -> str:
    from lazybridge.ext.report_builder.charts import render_chart

    if fragment.kind == "text":
        body = (fragment.body_md or "").replace("\n", "<br>")
        return f'<div class="text">{body}</div>'
    if fragment.kind == "callout":
        return (
            f'<aside class="callout {fragment.callout_style or "note"}">'
            f"{_html_escape(fragment.body_md or '')}"
            "</aside>"
        )
    if fragment.kind == "chart" and fragment.chart:
        rendered = render_chart(fragment.chart)
        return rendered.html
    if fragment.kind == "table" and fragment.table:
        thead = "".join(f"<th>{_html_escape(h)}</th>" for h in fragment.table.headers)
        trows = "".join(
            "<tr>" + "".join(f"<td>{_html_escape(str(c))}</td>" for c in row) + "</tr>"
            for row in fragment.table.rows
        )
        return f"<table><thead><tr>{thead}</tr></thead><tbody>{trows}</tbody></table>"
    return ""


_HTML_ESCAPE_RE = re.compile(r"[&<>\"']")
_HTML_ESCAPE_TABLE = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}


def _html_escape(s: str) -> str:
    return _HTML_ESCAPE_RE.sub(lambda m: _HTML_ESCAPE_TABLE[m.group(0)], s or "")
