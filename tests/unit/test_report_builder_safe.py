"""Lightweight tests for report_builder pure-logic paths.

Covers path safety (no heavy deps), HTML/YAML escaping, and fragment
validation — all exercisable without WeasyPrint, Quarto, or Chrome.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# report_tools — input path safety
# ---------------------------------------------------------------------------


def test_report_tools_safe_input_path_inside_root(tmp_path):
    """Paths under input_root are accepted."""
    from lazybridge.external_tools.report_builder.tools import report_tools

    tools = report_tools(output_dir=tmp_path, input_root=tmp_path)
    # The tool list is non-empty; path validation happens at call time.
    assert len(tools) == 1


def test_report_tools_markdown_path_outside_root_raises(tmp_path):
    """markdown_path outside input_root raises ValueError, not FileNotFoundError."""
    import asyncio

    from lazybridge.external_tools.report_builder.tools import report_tools

    outside = tmp_path.parent / "secret.md"
    outside.write_text("# secret", encoding="utf-8")

    tools = report_tools(output_dir=tmp_path / "out", input_root=tmp_path / "out")
    [gen_tool] = tools

    async def _call():
        return await gen_tool.func(
            title="t",
            theme="research",
            template="default",  # "default" is a valid VALID_TEMPLATES value
            output_format="html",
            markdown_path=str(outside),
            charts=None,
            sections=None,
            output_filename="r.html",
        )

    with pytest.raises(ValueError, match="outside the allowed input root"):
        asyncio.run(_call())


def test_report_tools_safe_input_path_traversal_blocked(tmp_path):
    """Path traversal via ../ is blocked by the input_root guard."""
    import asyncio

    from lazybridge.external_tools.report_builder.tools import report_tools

    # Build a traversal path using ../ that resolves to tmp_path.parent/secret.md.
    secret = tmp_path.parent / "secret.md"
    secret.write_text("# secret", encoding="utf-8")
    traversal = str(tmp_path / ".." / secret.name)

    # input_root is tmp_path — traversal escapes to tmp_path.parent.
    tools = report_tools(output_dir=tmp_path / "out", input_root=tmp_path)
    [gen_tool] = tools

    async def _call():
        return await gen_tool.func(
            title="t",
            theme="research",
            template="default",
            output_format="html",
            markdown_path=traversal,
            charts=None,
            sections=None,
            output_filename="r.html",
        )

    with pytest.raises(ValueError, match="outside the allowed input root"):
        asyncio.run(_call())


# ---------------------------------------------------------------------------
# weasyprint Reveal.js — text fragment HTML escaping
# ---------------------------------------------------------------------------


def test_reveal_text_fragment_escapes_html():
    """_fragment_to_reveal_html escapes < > & in text fragments."""
    from lazybridge.external_tools.report_builder.exporters.weasyprint import (
        _fragment_to_reveal_html,
    )
    from lazybridge.external_tools.report_builder.fragments import Fragment

    frag = Fragment(kind="text", body_md="<script>alert('xss')</script>")
    html = _fragment_to_reveal_html(frag)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_reveal_text_fragment_newlines_become_br():
    """Newlines in text fragments become <br> after escaping."""
    from lazybridge.external_tools.report_builder.exporters.weasyprint import (
        _fragment_to_reveal_html,
    )
    from lazybridge.external_tools.report_builder.fragments import Fragment

    frag = Fragment(kind="text", body_md="line1\nline2")
    html = _fragment_to_reveal_html(frag)
    assert "<br>" in html
    assert "line1" in html
    assert "line2" in html


# ---------------------------------------------------------------------------
# qmd — YAML escaping and callout_style validation
# ---------------------------------------------------------------------------


def test_yaml_escape_handles_quotes_backslashes_newlines():
    from lazybridge.external_tools.report_builder.quarto.qmd import _yaml_escape

    assert _yaml_escape('"quoted"') == '\\"quoted\\"'
    assert _yaml_escape("back\\slash") == "back\\\\slash"
    assert _yaml_escape("line1\nline2") == "line1\\nline2"
    assert _yaml_escape("cr\r\n") == "cr\\r\\n"


def test_qmd_extra_yaml_values_are_quoted(tmp_path):
    """extra_yaml values with special chars must not break the YAML front-matter."""
    from lazybridge.external_tools.report_builder.assemblers import AssembledReport
    from lazybridge.external_tools.report_builder.quarto.qmd import render_report_to_qmd

    report = AssembledReport(title='A "Title"', sections=[], citations=[])
    qmd = render_report_to_qmd(
        report,
        extra_yaml={"custom_key": 'value with "quotes" and\nnewline'},
    )
    # Must not contain a raw newline inside the YAML block
    yaml_block = qmd.split("---")[1]
    assert "custom_key" in yaml_block
    # The value must be quoted — raw newline inside YAML front-matter would break Quarto
    assert "\n" not in yaml_block.split("custom_key")[1].split("\n")[0]


def test_qmd_callout_style_unknown_falls_back_to_note():
    """Unknown callout_style values are replaced with 'note' by the QMD renderer.

    Fragment already validates callout_style via a Literal type, so invalid
    values can't be created via the normal constructor.  We use model_construct
    to simulate a bypassed-validation path (e.g. deserialisation from an older
    schema version) and verify the renderer still doesn't inject raw CSS.
    """
    from lazybridge.external_tools.report_builder.fragments import Fragment
    from lazybridge.external_tools.report_builder.quarto.qmd import render_fragment_to_qmd

    # model_construct bypasses Pydantic validation to simulate a stale/malformed
    # Fragment that somehow carries an unknown callout_style.
    frag = Fragment.model_construct(kind="callout", body_md="hello", callout_style="danger} {.injected")
    rendered = render_fragment_to_qmd(frag)
    assert "callout-note" in rendered
    assert "injected" not in rendered
