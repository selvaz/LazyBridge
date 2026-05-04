"""Core rendering logic for report_builder.

Functions lifted verbatim from the sandbox (markdown_to_clean_html,
build_html_document, escape_html, fallback_css, ALLOWED_TAGS, ALLOWED_ATTRS)
plus the new inject_charts_into_html and load_theme_css (adapted for
importlib.resources so it works in both editable and wheel installs),
and the Jinja2-based document builder with TOC extraction.
"""

from __future__ import annotations

import base64
import re
from importlib.resources import files as _resource_files
from pathlib import Path
from typing import Any

from lazybridge.ext.report_builder._deps import require_bleach, require_jinja2, require_markdown

# ---------------------------------------------------------------------------
# bleach whitelist — lifted verbatim from sandbox
# ---------------------------------------------------------------------------

ALLOWED_TAGS = [
    "p",
    "br",
    "hr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "strong",
    "em",
    "code",
    "pre",
    "blockquote",
    "ul",
    "ol",
    "li",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "a",
    "img",
    "div",
    "span",
    "figure",
    "figcaption",
]

ALLOWED_ATTRS = {
    "*": ["class", "id"],
    "a": ["href", "title"],
    "img": ["src", "alt", "title", "width", "height"],
    "th": ["align"],
    "td": ["align"],
    "code": ["class"],
    "pre": ["class"],
    "div": ["class"],
    "span": ["class"],
}

VALID_THEMES = frozenset({"executive", "financial", "technical", "research"})
VALID_TEMPLATES = frozenset({"default", "executive_summary", "deep_dive", "data_snapshot"})

# ---------------------------------------------------------------------------
# Markdown → sanitised HTML — lifted verbatim from sandbox
# ---------------------------------------------------------------------------


def markdown_to_clean_html(markdown_text: str) -> str:
    md = require_markdown()
    bleach = require_bleach()

    raw_html = md.markdown(
        markdown_text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
            "attr_list",
            "admonition",
        ],
        output_format="html5",
    )

    return bleach.clean(
        raw_html,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRS,
        protocols=["http", "https", "mailto"],
        strip=True,
    )


# ---------------------------------------------------------------------------
# HTML document builder — lifted verbatim from sandbox
# ---------------------------------------------------------------------------


def build_html_document(body_html: str, title: str, css: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape_html(title)}</title>
  <style>{css}</style>
</head>
<body>
  <main class="report-shell">
    <article class="markdown-body">
{body_html}
    </article>
  </main>
</body>
</html>
"""


def escape_html(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def fallback_css() -> str:
    return """
body {
  margin: 0;
  background: #f6f8fa;
  color: #24292f;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.65;
}

.report-shell {
  max-width: 1000px;
  margin: 40px auto;
  padding: 56px 64px;
  background: #ffffff;
  border: 1px solid #d0d7de;
  border-radius: 18px;
}

.markdown-body img {
  max-width: 100%;
}

.markdown-body table {
  display: block;
  max-width: 100%;
  overflow: auto;
  border-collapse: collapse;
}

.markdown-body th,
.markdown-body td {
  padding: 8px 12px;
  border: 1px solid #d0d7de;
}
"""


# ---------------------------------------------------------------------------
# Theme loader — adapted: importlib.resources instead of path-relative
# ---------------------------------------------------------------------------


def load_theme_css(theme: str) -> str:
    """Load CSS for one of the four built-in themes.

    Args:
        theme: One of ``executive``, ``financial``, ``technical``, ``research``.

    Raises:
        ValueError: If *theme* is not in VALID_THEMES.
    """
    if theme not in VALID_THEMES:
        raise ValueError(f"Unknown theme {theme!r}. Choose from: {sorted(VALID_THEMES)}")

    pkg = _resource_files("lazybridge.ext.report_builder.themes")
    return (pkg / f"{theme}.css").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Chart helpers — new in report_builder (not in sandbox)
# ---------------------------------------------------------------------------


def _word_overlap(a: str, b: str) -> float:
    """Fraction of *a*'s words that appear in *b* (case-insensitive)."""
    wa = set(re.sub(r"[^\w\s]", "", a.lower()).split())
    wb = set(re.sub(r"[^\w\s]", "", b.lower()).split())
    return len(wa & wb) / len(wa) if wa else 0.0


def _chart_to_figure_html(chart_path: Path, title: str) -> str:
    """Read a PNG file and return an HTML <figure> block with an inline data URI."""
    b64 = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    safe_title = escape_html(title)
    return (
        f"<figure>\n"
        f'  <img src="data:image/png;base64,{b64}" alt="{safe_title}" style="max-width:100%;" />\n'
        f"  <figcaption>{safe_title}</figcaption>\n"
        f"</figure>"
    )


def inject_charts_into_html(body_html: str, chart_items: list[tuple[str, str]]) -> str:
    """Insert chart <figure> blocks into *body_html* using heading-name matching.

    Each item in *chart_items* is a ``(match_name, figure_html)`` pair.
    The function scores every ``<h1>/<h2>/<h3>`` heading in *body_html* against
    *match_name* using word-overlap and inserts the figure immediately after the
    best-matching heading.  Charts with no heading match are appended at the end.

    Args:
        body_html: Sanitised HTML body produced by markdown_to_clean_html.
        chart_items: List of ``(name, figure_html)`` pairs.

    Returns:
        Modified HTML body with chart figures inserted.
    """
    heading_pattern = re.compile(r"(<h[1-3][^>]*>)(.*?)(</h[1-3]>)", re.IGNORECASE | re.DOTALL)
    headings = [(m.group(2), m.end()) for m in heading_pattern.finditer(body_html)]

    unmatched: list[str] = []
    insertions: list[tuple[int, str]] = []  # (char_pos, figure_html)

    for name, figure_html in chart_items:
        best_score, best_pos = 0.0, -1
        for heading_text, end_pos in headings:
            score = _word_overlap(name, heading_text)
            if score > best_score:
                best_score, best_pos = score, end_pos
        if best_pos >= 0:
            insertions.append((best_pos, figure_html))
        else:
            unmatched.append(figure_html)

    # Apply in reverse order so earlier insertions don't shift later positions
    for pos, html in sorted(insertions, key=lambda x: x[0], reverse=True):
        body_html = body_html[:pos] + "\n" + html + "\n" + body_html[pos:]

    if unmatched:
        body_html += "\n" + "\n".join(unmatched)

    return body_html


# ---------------------------------------------------------------------------
# TOC extraction
# ---------------------------------------------------------------------------


def extract_toc(body_html: str) -> list[dict[str, Any]]:
    """Extract h2/h3 headings from *body_html* for use in the deep_dive TOC.

    Returns a list of dicts with keys ``level``, ``text``, ``id``.
    The ``id`` is taken from an existing ``id="..."`` attribute if present,
    otherwise derived by slugifying the text.
    """
    pattern = re.compile(
        r'<h([23])(?:[^>]*\bid="([^"]*)")?[^>]*>(.*?)</h\1>',
        re.IGNORECASE | re.DOTALL,
    )
    items = []
    for m in pattern.finditer(body_html):
        level = int(m.group(1))
        existing_id = m.group(2) or ""
        raw_text = re.sub(r"<[^>]+>", "", m.group(3)).strip()
        _id = existing_id if existing_id else re.sub(r"[^\w-]", "-", raw_text.lower()).strip("-")
        if raw_text:
            items.append({"level": level, "text": raw_text, "id": _id})
    return items


# ---------------------------------------------------------------------------
# Jinja2-based document builder
# ---------------------------------------------------------------------------


def build_html_document_jinja2(
    body_html: str,
    title: str,
    css: str,
    template_name: str,
    meta: dict[str, Any],
) -> str:
    """Render the final HTML document using a named Jinja2 template.

    Args:
        body_html: Sanitised HTML body (from markdown or section renderer).
        title: Report title (injected into <title> and heading elements).
        css: Full CSS string from load_theme_css().
        template_name: One of ``default``, ``executive_summary``,
            ``deep_dive``, ``data_snapshot``.
        meta: Dict with keys ``generated_at``, ``theme``,
            ``charts_embedded``, ``template``.

    Returns:
        Complete HTML document as a string.
    """
    if template_name not in VALID_TEMPLATES:
        raise ValueError(f"Unknown template {template_name!r}. Choose from: {sorted(VALID_TEMPLATES)}")

    jinja2 = require_jinja2()

    pkg = _resource_files("lazybridge.ext.report_builder.templates")
    template_str = (pkg / f"{template_name}.html.j2").read_text(encoding="utf-8")

    env = jinja2.Environment(autoescape=False)  # body_html is pre-sanitised
    tmpl = env.from_string(template_str)

    toc_items = extract_toc(body_html) if template_name == "deep_dive" else []

    return tmpl.render(
        body_html=body_html,
        title=title,
        css=css,
        toc_items=toc_items,
        meta=meta,
    )
