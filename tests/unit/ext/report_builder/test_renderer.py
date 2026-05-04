"""Unit tests for report_builder renderer functions.

Heavy deps (markdown, bleach, jinja2) are skipped if not installed.
"""

from __future__ import annotations

import pytest

markdown_mod = pytest.importorskip("markdown")
bleach_mod = pytest.importorskip("bleach")
jinja2_mod = pytest.importorskip("jinja2")

from lazybridge.ext.report_builder.renderer import (
    VALID_TEMPLATES,
    VALID_THEMES,
    _word_overlap,
    build_html_document,
    build_html_document_jinja2,
    escape_html,
    extract_toc,
    inject_charts_into_html,
    load_theme_css,
    markdown_to_clean_html,
)

# ---------------------------------------------------------------------------
# escape_html
# ---------------------------------------------------------------------------


class TestEscapeHtml:
    def test_ampersand(self):
        assert escape_html("A & B") == "A &amp; B"

    def test_angle_brackets(self):
        assert escape_html("<script>") == "&lt;script&gt;"

    def test_quotes(self):
        assert '"hello"' in escape_html('"hello"') or "&quot;" in escape_html('"hello"')

    def test_plain_string_unchanged(self):
        assert escape_html("Hello World") == "Hello World"


# ---------------------------------------------------------------------------
# markdown_to_clean_html
# ---------------------------------------------------------------------------


class TestMarkdownToCleanHtml:
    def test_heading_rendered(self):
        html = markdown_to_clean_html("# Title")
        assert "<h1" in html  # toc extension adds id="..." attribute
        assert "Title" in html

    def test_script_tag_stripped(self):
        html = markdown_to_clean_html("<script>alert('xss')</script>")
        assert "<script>" not in html

    def test_inline_code_preserved(self):
        html = markdown_to_clean_html("Use `print()` here")
        assert "<code>" in html
        assert "print()" in html

    def test_table_rendered(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html = markdown_to_clean_html(md)
        assert "<table>" in html
        assert "<td>" in html

    def test_strong_preserved(self):
        html = markdown_to_clean_html("**bold**")
        assert "<strong>" in html

    def test_returns_string(self):
        result = markdown_to_clean_html("hello")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_html_document (legacy, kept for backwards compat)
# ---------------------------------------------------------------------------


class TestBuildHtmlDocument:
    def test_doctype_present(self):
        html = build_html_document("<p>body</p>", "T", "body{}")
        assert html.strip().lower().startswith("<!doctype html>")

    def test_title_included(self):
        html = build_html_document("<p>x</p>", "My Report", "")
        assert "My Report" in html

    def test_title_escaped(self):
        html = build_html_document("<p>x</p>", "<script>evil</script>", "")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_css_embedded(self):
        html = build_html_document("<p>x</p>", "T", "body{color:red;}")
        assert "body{color:red;}" in html

    def test_body_html_included(self):
        html = build_html_document("<p>hello world</p>", "T", "")
        assert "<p>hello world</p>" in html

    def test_report_shell_class(self):
        html = build_html_document("", "T", "")
        assert "report-shell" in html


# ---------------------------------------------------------------------------
# load_theme_css
# ---------------------------------------------------------------------------


class TestLoadThemeCss:
    @pytest.mark.parametrize("theme", sorted(VALID_THEMES))
    def test_all_themes_load(self, theme):
        css = load_theme_css(theme)
        assert isinstance(css, str)
        assert len(css) > 100

    @pytest.mark.parametrize("theme", sorted(VALID_THEMES))
    def test_themes_contain_report_shell(self, theme):
        css = load_theme_css(theme)
        assert ".report-shell" in css

    def test_unknown_theme_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown theme"):
            load_theme_css("neon")

    def test_unknown_theme_lists_valid_options(self):
        with pytest.raises(ValueError, match="executive"):
            load_theme_css("bogus")


# ---------------------------------------------------------------------------
# _word_overlap
# ---------------------------------------------------------------------------


class TestWordOverlap:
    def test_exact_match(self):
        assert _word_overlap("revenue by quarter", "Revenue by Quarter") == 1.0

    def test_partial_match(self):
        score = _word_overlap("revenue", "Revenue by Quarter")
        assert 0.0 < score <= 1.0

    def test_no_match(self):
        assert _word_overlap("unrelated topic", "Revenue by Quarter") == 0.0

    def test_empty_a_returns_zero(self):
        assert _word_overlap("", "Revenue") == 0.0

    def test_punctuation_ignored(self):
        assert _word_overlap("revenue, analysis", "Revenue Analysis") == 1.0


# ---------------------------------------------------------------------------
# inject_charts_into_html
# ---------------------------------------------------------------------------


class TestInjectChartsIntoHtml:
    def _html(self, text: str) -> str:
        return markdown_to_clean_html(text)

    def test_chart_inserted_after_matching_heading(self):
        html = self._html("## Revenue\n\nSome text.")
        figure = '<figure><img src="x.png" /></figure>'
        result = inject_charts_into_html(html, [("Revenue", figure)])
        h2_pos = result.index("</h2>")
        fig_pos = result.index("<figure>")
        assert fig_pos > h2_pos

    def test_unmatched_chart_appended_at_end(self):
        html = self._html("## Revenue\n\nText.")
        figure = '<figure><img src="x.png" /></figure>'
        result = inject_charts_into_html(html, [("Unrelated Topic XYZ", figure)])
        assert figure in result

    def test_multiple_charts_placed_at_respective_headings(self):
        html = self._html("## Revenue\n\nText.\n\n## Segment\n\nOther text.")
        fig_rev = "<figure>rev</figure>"
        fig_seg = "<figure>seg</figure>"
        result = inject_charts_into_html(html, [("Revenue", fig_rev), ("Segment", fig_seg)])
        assert result.index(fig_rev) < result.index(fig_seg)

    def test_no_charts_returns_body_unchanged(self):
        html = self._html("## Section\n\nContent.")
        result = inject_charts_into_html(html, [])
        assert result == html

    def test_returns_string(self):
        result = inject_charts_into_html("<p>x</p>", [("name", "<figure/>")])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extract_toc
# ---------------------------------------------------------------------------


class TestExtractToc:
    def test_extracts_h2_items(self):
        html = '<h2 id="rev">Revenue</h2><p>text</p><h2 id="seg">Segments</h2>'
        toc = extract_toc(html)
        assert len(toc) == 2
        assert toc[0] == {"level": 2, "text": "Revenue", "id": "rev"}
        assert toc[1] == {"level": 2, "text": "Segments", "id": "seg"}

    def test_extracts_h3_items(self):
        html = '<h2 id="a">Section</h2><h3 id="b">Subsection</h3>'
        toc = extract_toc(html)
        assert any(t["level"] == 3 for t in toc)

    def test_slugifies_id_when_missing(self):
        html = "<h2>Revenue by Quarter</h2>"
        toc = extract_toc(html)
        assert len(toc) == 1
        assert "revenue" in toc[0]["id"]

    def test_h1_not_included(self):
        html = "<h1>Title</h1><h2>Section</h2>"
        toc = extract_toc(html)
        assert all(t["level"] != 1 for t in toc)

    def test_empty_html_returns_empty(self):
        assert extract_toc("<p>no headings</p>") == []


# ---------------------------------------------------------------------------
# build_html_document_jinja2
# ---------------------------------------------------------------------------


class TestBuildHtmlDocumentJinja2:
    @pytest.mark.parametrize("tmpl", sorted(VALID_TEMPLATES))
    def test_all_templates_render(self, tmpl):
        meta = {"generated_at": "2026-01-01", "theme": "executive", "template": tmpl, "charts_embedded": 0}
        html = build_html_document_jinja2("<p>body</p>", "Test", "body{}", tmpl, meta)
        assert "<!doctype html>" in html.lower()
        assert "Test" in html
        assert "<p>body</p>" in html

    def test_default_template_has_report_shell(self):
        meta = {"generated_at": "2026-01-01", "theme": "executive", "template": "default", "charts_embedded": 0}
        html = build_html_document_jinja2("<p>x</p>", "T", "", "default", meta)
        assert "report-shell" in html

    def test_executive_summary_has_meta_chip(self):
        meta = {
            "generated_at": "2026-05-01",
            "theme": "financial",
            "template": "executive_summary",
            "charts_embedded": 2,
        }
        html = build_html_document_jinja2("<p>x</p>", "T", "", "executive_summary", meta)
        assert "meta-chip" in html
        assert "2026-05-01" in html
        assert "2 charts" in html

    def test_deep_dive_has_toc_when_headings_present(self):
        body = '<h2 id="rev">Revenue</h2><p>text</p>'
        meta = {"generated_at": "2026-01-01", "theme": "executive", "template": "deep_dive", "charts_embedded": 0}
        html = build_html_document_jinja2(body, "T", "", "deep_dive", meta)
        assert "dd-toc" in html
        assert "Revenue" in html

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            build_html_document_jinja2("<p>x</p>", "T", "", "bogus_template", {})
