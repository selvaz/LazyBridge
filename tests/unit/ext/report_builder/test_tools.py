"""Unit tests for report_tools factory and generate_report tool."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from lazybridge import Tool
from lazybridge.ext.report_builder.tools import report_tools

# ---------------------------------------------------------------------------
# Minimal valid PNG bytes (1×1 pixel)
# ---------------------------------------------------------------------------

_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
    b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd4n\x00\x00\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------------
# Factory contract
# ---------------------------------------------------------------------------


class TestReportToolsFactory:
    def test_returns_list(self, tmp_path):
        tools = report_tools(tmp_path)
        assert isinstance(tools, list)

    def test_returns_one_tool(self, tmp_path):
        assert len(report_tools(tmp_path)) == 1

    def test_tool_is_lazybridge_tool(self, tmp_path):
        assert isinstance(report_tools(tmp_path)[0], Tool)

    def test_tool_name(self, tmp_path):
        assert report_tools(tmp_path)[0].name == "generate_report"

    def test_tool_has_description(self, tmp_path):
        tool = report_tools(tmp_path)[0]
        assert tool.description and len(tool.description) > 10

    def test_schema_has_expected_params(self, tmp_path):
        props = report_tools(tmp_path)[0].definition().parameters["properties"]
        for param in (
            "title",
            "theme",
            "template",
            "sections",
            "markdown_path",
            "charts",
            "output_filename",
            "output_format",
        ):
            assert param in props, f"Missing param: {param}"

    def test_output_dir_created_on_first_call(self, tmp_path):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")
        pytest.importorskip("jinja2")

        new_dir = tmp_path / "nested" / "reports"
        md = tmp_path / "test.md"
        md.write_text("# Hello", encoding="utf-8")

        result = report_tools(new_dir)[0].func(markdown_path=str(md), title="Test")
        assert not result.get("error")
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# Input validation errors (no markdown/bleach needed for most)
# ---------------------------------------------------------------------------


class TestGenerateReportErrors:
    def test_no_content_source_returns_error(self, tmp_path):
        result = report_tools(tmp_path)[0].func(title="T")
        assert result["error"] is True
        assert "sections" in result["message"] or "markdown_path" in result["message"]

    def test_missing_markdown_file_returns_error(self, tmp_path):
        result = report_tools(tmp_path)[0].func(markdown_path=str(tmp_path / "nonexistent.md"), title="T")
        assert result["error"] is True
        assert result["type"] == "FileNotFoundError"

    def test_unknown_theme_returns_error(self, tmp_path):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")

        md = tmp_path / "t.md"
        md.write_text("# Hi", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", theme="neon")
        assert result["error"] is True
        assert "neon" in result["message"]

    def test_unknown_template_returns_error(self, tmp_path):
        md = tmp_path / "t.md"
        md.write_text("# Hi", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", template="bogus")
        assert result["error"] is True
        assert "bogus" in result["message"]

    def test_unknown_output_format_returns_error(self, tmp_path):
        md = tmp_path / "t.md"
        md.write_text("# Hi", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", output_format="docx")
        assert result["error"] is True

    def test_invalid_chart_spec_returns_error(self, tmp_path):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")

        md = tmp_path / "t.md"
        md.write_text("# Hi", encoding="utf-8")
        # missing required 'title' and 'path' fields
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", charts=[{"name": "x"}])
        assert result["error"] is True

    def test_missing_chart_png_returns_error(self, tmp_path):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")

        md = tmp_path / "t.md"
        md.write_text("# Hi", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(
            markdown_path=str(md),
            title="T",
            charts=[{"path": str(tmp_path / "missing.png"), "title": "Chart", "name": "Hi"}],
        )
        assert result["error"] is True
        assert result["type"] == "FileNotFoundError"

    def test_invalid_section_type_returns_error(self, tmp_path):
        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[{"type": "video", "url": "x"}],
        )
        assert result["error"] is True
        assert "video" in result["message"]

    def test_invalid_section_fields_returns_error(self, tmp_path):
        # ChartSection requires 'path' and 'title'
        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[{"type": "chart"}],
        )
        assert result["error"] is True


# ---------------------------------------------------------------------------
# markdown_path flow — success (requires markdown + bleach + jinja2)
# ---------------------------------------------------------------------------


class TestMarkdownPathFlow:
    def setup_method(self):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")
        pytest.importorskip("jinja2")

    def test_minimal_report_created(self, tmp_path):
        md = tmp_path / "report.md"
        md.write_text("# Hello\n\nThis is a test report.", encoding="utf-8")

        result = report_tools(tmp_path / "out")[0].func(markdown_path=str(md), title="Test Report")

        assert not result.get("error"), result.get("message")
        assert result["title"] == "Test Report"
        assert result["charts_embedded"] == 0
        assert result["theme"] == "executive"
        assert result["template"] == "default"
        assert Path(result["html_path"]).exists()

    def test_output_is_valid_html(self, tmp_path):
        md = tmp_path / "r.md"
        md.write_text("# Section\n\nContent.", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T")

        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "<!doctype html>" in html.lower()
        assert "<title>" in html
        assert "report-shell" in html

    def test_all_themes_produce_output(self, tmp_path):
        md = tmp_path / "r.md"
        md.write_text("# Test", encoding="utf-8")

        for theme in ("executive", "financial", "technical", "research"):
            result = report_tools(tmp_path)[0].func(
                markdown_path=str(md),
                title="T",
                theme=theme,
                output_filename=f"{theme}.html",
            )
            assert not result.get("error"), f"Theme {theme!r}: {result}"
            assert result["theme"] == theme

    def test_chart_embedded_as_base64(self, tmp_path):
        png_path = tmp_path / "chart.png"
        png_path.write_bytes(_PNG)

        md = tmp_path / "r.md"
        md.write_text("# Revenue\n\nText.", encoding="utf-8")

        result = report_tools(tmp_path)[0].func(
            markdown_path=str(md),
            title="T",
            charts=[{"path": str(png_path), "title": "Revenue Chart", "name": "Revenue"}],
        )

        assert not result.get("error"), result.get("message")
        assert result["charts_embedded"] == 1
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "data:image/png;base64," in html
        assert base64.b64encode(_PNG).decode("ascii") in html

    def test_custom_output_filename(self, tmp_path):
        md = tmp_path / "r.md"
        md.write_text("# T", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", output_filename="custom_name.html")
        assert not result.get("error")
        assert Path(result["html_path"]).name == "custom_name.html"


# ---------------------------------------------------------------------------
# sections flow — success (requires markdown + bleach + jinja2)
# ---------------------------------------------------------------------------


class TestSectionsFlow:
    def setup_method(self):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")
        pytest.importorskip("jinja2")

    def test_text_section_renders(self, tmp_path):
        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[{"type": "text", "heading": "Overview", "body": "Some **bold** text."}],
        )
        assert not result.get("error"), result.get("message")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "Overview" in html
        assert "<strong>" in html

    def test_table_section_renders(self, tmp_path):
        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[
                {
                    "type": "table",
                    "caption": "Sales Summary",
                    "headers": ["Region", "Revenue"],
                    "rows": [["EMEA", "1.2M"], ["APAC", "0.9M"]],
                }
            ],
        )
        assert not result.get("error"), result.get("message")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "<table>" in html
        assert "EMEA" in html
        assert "Sales Summary" in html

    def test_chart_section_embedded(self, tmp_path):
        png_path = tmp_path / "c.png"
        png_path.write_bytes(_PNG)

        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[{"type": "chart", "path": str(png_path), "title": "My Chart"}],
        )
        assert not result.get("error"), result.get("message")
        assert result["charts_embedded"] == 1
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "data:image/png;base64," in html

    def test_sections_charts_ignored_for_placement(self, tmp_path):
        png_path = tmp_path / "c.png"
        png_path.write_bytes(_PNG)

        # charts param is ignored when sections is provided
        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[{"type": "chart", "path": str(png_path), "title": "My Chart"}],
            charts=[{"path": str(png_path), "title": "Ignored", "name": "x"}],
        )
        # should succeed without error; charts param is silently ignored
        assert not result.get("error"), result.get("message")
        assert result["charts_embedded"] == 1

    def test_mixed_sections_order_preserved(self, tmp_path):
        png_path = tmp_path / "c.png"
        png_path.write_bytes(_PNG)

        result = report_tools(tmp_path)[0].func(
            title="T",
            sections=[
                {"type": "text", "heading": "Intro", "body": "Hello."},
                {"type": "chart", "path": str(png_path), "title": "Chart A"},
                {"type": "table", "headers": ["K", "V"], "rows": [["a", "1"]]},
            ],
        )
        assert not result.get("error"), result.get("message")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        intro_pos = html.index("Intro")
        chart_pos = html.index("data:image/png")
        table_pos = html.index("<table>")
        assert intro_pos < chart_pos < table_pos


# ---------------------------------------------------------------------------
# Template variants (requires markdown + bleach + jinja2)
# ---------------------------------------------------------------------------


class TestTemplateVariants:
    def setup_method(self):
        pytest.importorskip("markdown")
        pytest.importorskip("bleach")
        pytest.importorskip("jinja2")

    @pytest.mark.parametrize("template", ["default", "executive_summary", "deep_dive", "data_snapshot"])
    def test_all_templates_produce_valid_html(self, tmp_path, template):
        md = tmp_path / "r.md"
        md.write_text("## Revenue\n\nText.\n\n## Segments\n\nMore.", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(
            markdown_path=str(md),
            title="T",
            template=template,
            output_filename=f"{template}.html",
        )
        assert not result.get("error"), f"Template {template!r}: {result}"
        assert result["template"] == template
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "<!doctype html>" in html.lower()

    def test_deep_dive_contains_toc(self, tmp_path):
        md = tmp_path / "r.md"
        md.write_text("## Revenue\n\nText.\n\n## Segments\n\nMore.", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", template="deep_dive")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "dd-toc" in html

    def test_executive_summary_contains_meta_chip(self, tmp_path):
        md = tmp_path / "r.md"
        md.write_text("# Report\n\nContent.", encoding="utf-8")
        result = report_tools(tmp_path)[0].func(markdown_path=str(md), title="T", template="executive_summary")
        html = Path(result["html_path"]).read_text(encoding="utf-8")
        assert "meta-chip" in html
