"""Renderer for typed content sections (TextSection, ChartSection, TableSection).

This module converts a ``list[ReportSection]`` into a single HTML body string
without touching the Markdown → HTML pipeline used by the ``markdown_path`` flow.

Sections are rendered **in order**, so chart placement is explicit and
deterministic — no word-overlap heuristics needed.
"""

from __future__ import annotations

import re
from pathlib import Path

from lazybridge.ext.report_builder.renderer import (
    _chart_to_figure_html,
    escape_html,
    markdown_to_clean_html,
)
from lazybridge.ext.report_builder.schemas import ChartSection, TableSection, TextSection


def _slugify(text: str) -> str:
    """Convert a heading string to a URL-safe id attribute."""
    slug = re.sub(r"[^\w\s-]", "", text.lower().strip())
    return re.sub(r"[\s_-]+", "-", slug).strip("-") or "section"


def render_text_section(section: TextSection) -> str:
    """Render a TextSection to HTML."""
    parts: list[str] = []
    if section.heading.strip():
        _id = _slugify(section.heading)
        parts.append(f'<h2 id="{_id}">{escape_html(section.heading)}</h2>')
    if section.body.strip():
        parts.append(markdown_to_clean_html(section.body))
    return "\n".join(parts)


def render_chart_section(section: ChartSection) -> str:
    """Render a ChartSection to HTML — reads PNG, encodes base64."""
    parts: list[str] = []
    if section.heading.strip():
        _id = _slugify(section.heading)
        parts.append(f'<h3 id="{_id}">{escape_html(section.heading)}</h3>')
    chart_path = Path(section.path)
    if not chart_path.exists():
        raise FileNotFoundError(f"Chart image not found: {section.path}")
    parts.append(_chart_to_figure_html(chart_path, section.title))
    return "\n".join(parts)


def render_table_section(section: TableSection) -> str:
    """Render a TableSection to an HTML <table>."""
    parts: list[str] = []
    if section.caption.strip():
        parts.append(f"<p><strong>{escape_html(section.caption)}</strong></p>")
    parts.append("<table>")
    parts.append("<thead><tr>")
    for h in section.headers:
        parts.append(f"<th>{escape_html(h)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in section.rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{escape_html(str(cell))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def sections_to_html(sections: list) -> tuple[str, int]:
    """Convert a list of ReportSection dicts/models to HTML body + chart count.

    Returns:
        (body_html, charts_embedded)
    """
    parts: list[str] = []
    charts = 0
    for sec in sections:
        if isinstance(sec, TextSection):
            parts.append(render_text_section(sec))
        elif isinstance(sec, ChartSection):
            parts.append(render_chart_section(sec))
            charts += 1
        elif isinstance(sec, TableSection):
            parts.append(render_table_section(sec))
    return "\n".join(parts), charts
