"""Tool factory for report_builder.

Closure-based factory — each call to ``report_tools(output_dir)`` returns a
``generate_report`` tool permanently bound to that output directory.

Usage::

    from lazybridge import Agent
    from lazybridge.ext.report_builder import report_tools

    agent = Agent("anthropic", tools=report_tools("./reports"))
    agent(
        "Generate a Q1 analysis report. Use analysis.md and the charts "
        "in ./output/charts/ for the Revenue and Segment sections."
    )
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

from lazybridge import Tool


def _error(exc: Exception) -> dict[str, Any]:
    return {"error": True, "type": type(exc).__name__, "message": str(exc)}


def report_tools(output_dir: str | Path) -> list[Tool]:
    """Return a list containing the ``generate_report`` tool bound to *output_dir*.

    Args:
        output_dir: Directory where report files will be written.
                    Created automatically if it does not exist.

    Returns:
        A single-element list: ``[generate_report tool]``.

    Example::

        from lazybridge import Agent
        from lazybridge.ext.report_builder import report_tools

        agent = Agent("anthropic", tools=report_tools("./reports"))
        agent("Assemble the quarterly report from analysis.md and the chart PNGs.")
    """
    _out = Path(output_dir).resolve()

    # ------------------------------------------------------------------
    # generate_report
    # ------------------------------------------------------------------

    def generate_report(
        title: Annotated[str, "Report title — shown in the browser tab and document heading"],
        theme: Annotated[
            str,
            "Visual theme: 'executive' (dark-blue, corporate), 'financial' (teal, tabular), "
            "'technical' (dark mode, monospace), 'research' (serif, academic). Default: executive.",
        ] = "executive",
        template: Annotated[
            str,
            "Layout template: "
            "'default' (clean single-column), "
            "'executive_summary' (header bar with meta chips + footer), "
            "'deep_dive' (sticky TOC sidebar + full-width content), "
            "'data_snapshot' (chart-dominant, compact text). Default: default.",
        ] = "default",
        sections: Annotated[
            list[dict] | None,
            "Typed content sections — alternative to markdown_path. "
            "Each dict must have a 'type' key: "
            "'text' ({type, heading?, body}), "
            "'chart' ({type, path, title, heading?}), "
            "'table' ({type, caption?, headers, rows}). "
            "Sections are rendered in order — no auto-placement needed. "
            "Use this when the LLM generates content directly without writing a file.",
        ] = None,
        markdown_path: Annotated[
            str | None,
            "Path to a .md file produced by upstream tools. "
            "Used when content already exists on disk. "
            "Mutually exclusive with 'sections' (sections takes priority if both are provided).",
        ] = None,
        charts: Annotated[
            list[dict] | None,
            "Chart refs for the markdown_path flow only (ignored when sections is provided). "
            "Each dict: 'path' (PNG file path), 'title' (caption), 'name' (matches section heading). "
            "Charts are auto-placed after the best-matching heading via word-overlap scoring.",
        ] = None,
        output_filename: Annotated[str, "Output filename, e.g. 'report.html' or 'q1_2026.html'"] = "report.html",
        output_format: Annotated[
            str,
            "Output format: 'html' (default), 'pdf' (requires pip install lazybridge[pdf]), "
            "or 'both' (produces both files).",
        ] = "html",
    ) -> dict[str, Any]:
        """Assemble a self-contained report from typed sections or a Markdown file plus chart PNGs.

        Two input modes:
        - **sections** mode: pass a list of typed dicts (text/chart/table) directly.
          Charts are embedded in order — no heading-matching needed.
        - **markdown_path** mode: pass a path to an existing .md file.
          Charts are placed inline via heading word-overlap auto-placement.

        All PNG charts are base64-encoded and embedded as data URIs — the output
        HTML is fully self-contained (no external files required to view it).

        Returns a dict with html_path and/or pdf_path, plus title, charts_embedded,
        theme, and template.  On failure returns {"error": True, "type": ..., "message": ...}.
        """
        try:
            from lazybridge.ext.report_builder.renderer import (
                VALID_TEMPLATES,
                VALID_THEMES,
                _chart_to_figure_html,
                build_html_document_jinja2,
                inject_charts_into_html,
                load_theme_css,
                markdown_to_clean_html,
            )
            from lazybridge.ext.report_builder.schemas import (
                ChartRef,
                ChartSection,
                TableSection,
                TextSection,
            )
            from lazybridge.ext.report_builder.section_renderer import sections_to_html

            # ── validate theme / template ─────────────────────────────
            if theme not in VALID_THEMES:
                return _error(ValueError(
                    f"Unknown theme {theme!r}. Choose from: {sorted(VALID_THEMES)}"
                ))
            if template not in VALID_TEMPLATES:
                return _error(ValueError(
                    f"Unknown template {template!r}. Choose from: {sorted(VALID_TEMPLATES)}"
                ))

            # ── validate output_format ────────────────────────────────
            valid_formats = {"html", "pdf", "both"}
            if output_format not in valid_formats:
                return _error(ValueError(
                    f"Unknown output_format {output_format!r}. Choose from: {sorted(valid_formats)}"
                ))

            # ── require at least one content source ───────────────────
            if sections is None and markdown_path is None:
                return _error(ValueError(
                    "Provide either 'sections' (typed content blocks) "
                    "or 'markdown_path' (path to a .md file)."
                ))

            # ── validate output path ──────────────────────────────────
            _out.mkdir(parents=True, exist_ok=True)
            safe_base = Path(output_filename).name
            if not safe_base.endswith(".html"):
                safe_base += ".html"
            html_path = (_out / safe_base).resolve()
            try:
                html_path.relative_to(_out)
            except ValueError:
                return _error(ValueError(f"Unsafe output filename: {output_filename!r}"))

            # ── build body_html ───────────────────────────────────────
            charts_embedded = 0

            if sections is not None:
                # --- typed sections flow ---
                parsed_sections = []
                for i, s in enumerate(sections):
                    t = s.get("type", "")
                    try:
                        if t == "text":
                            parsed_sections.append(TextSection(**s))
                        elif t == "chart":
                            parsed_sections.append(ChartSection(**s))
                        elif t == "table":
                            parsed_sections.append(TableSection(**s))
                        else:
                            return _error(ValueError(
                                f"sections[{i}]: unknown type {t!r}. Use 'text', 'chart', or 'table'."
                            ))
                    except Exception as exc:
                        return _error(ValueError(f"sections[{i}] is invalid: {exc}"))

                body_html, charts_embedded = sections_to_html(parsed_sections)

            else:
                # --- markdown_path flow ---
                md_path = Path(markdown_path)
                if not md_path.exists():
                    return _error(FileNotFoundError(f"Markdown file not found: {markdown_path}"))
                markdown_text = md_path.read_text(encoding="utf-8")
                body_html = markdown_to_clean_html(markdown_text)

                chart_refs: list[ChartRef] = []
                if charts:
                    for i, c in enumerate(charts):
                        try:
                            chart_refs.append(ChartRef(**c))
                        except Exception as exc:
                            return _error(ValueError(f"charts[{i}] is invalid: {exc}"))

                chart_items: list[tuple[str, str]] = []
                for ref in chart_refs:
                    cp = Path(ref.path)
                    if not cp.exists():
                        return _error(FileNotFoundError(f"Chart image not found: {ref.path}"))
                    if not cp.is_file():
                        return _error(ValueError(f"Chart path is not a file: {ref.path}"))
                    chart_items.append((ref.match_name, _chart_to_figure_html(cp, ref.title)))
                    charts_embedded += 1

                if chart_items:
                    body_html = inject_charts_into_html(body_html, chart_items)

            # ── apply theme + template ────────────────────────────────
            css = load_theme_css(theme)
            meta = {
                "generated_at": date.today().isoformat(),
                "theme": theme,
                "template": template,
                "charts_embedded": charts_embedded,
            }
            final_html = build_html_document_jinja2(body_html, title, css, template, meta)

            # ── write outputs ─────────────────────────────────────────
            result: dict[str, Any] = {
                "title": title,
                "charts_embedded": charts_embedded,
                "theme": theme,
                "template": template,
            }

            if output_format in ("html", "both"):
                html_path.write_text(final_html, encoding="utf-8")
                result["html_path"] = str(html_path)

            if output_format in ("pdf", "both"):
                from lazybridge.ext.report_builder._deps import require_weasyprint
                weasyprint = require_weasyprint()
                pdf_path = html_path.with_suffix(".pdf")
                weasyprint.HTML(string=final_html).write_pdf(str(pdf_path))
                result["pdf_path"] = str(pdf_path)

            return result

        except Exception as exc:
            return _error(exc)

    return [
        Tool(
            generate_report,
            name="generate_report",
            description=(
                "Assemble a self-contained HTML (and optionally PDF) report from typed content sections "
                "or a Markdown file, with optional embedded chart PNG images. "
                "4 layout templates × 4 CSS themes. Charts auto-placed or ordered via sections."
            ),
            guidance=(
                "Two input modes: (1) 'sections' — pass typed text/chart/table dicts directly, "
                "charts embedded in order, no file needed; "
                "(2) 'markdown_path' — path to existing .md file with 'charts' for auto-placement. "
                "Themes: executive, financial, technical, research. "
                "Templates: default, executive_summary (meta header+footer), "
                "deep_dive (TOC sidebar), data_snapshot (chart-prominent). "
                "output_format: 'html' (default), 'pdf' (requires lazybridge[pdf]), 'both'. "
                "All chart PNGs are base64-embedded — output is fully self-contained."
            ),
        )
    ]
