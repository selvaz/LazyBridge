"""Smoke tests for the Quarto exporter — gated on the CLI being on PATH."""

from __future__ import annotations

import shutil

import pytest

from lazybridge.ext.report_builder import (
    BlackboardAssembler,
    Citation,
    FragmentBus,
)
from lazybridge.ext.report_builder.exporters.quarto import QuartoExporter
from lazybridge.ext.report_builder.fragments import (
    ChartSpec,
    Fragment,
    Provenance,
    TableSpec,
)
from lazybridge.ext.report_builder.quarto.detect import (
    QuartoNotFoundError,
    find_quarto,
)
from lazybridge.ext.report_builder.quarto.project import build_quarto_yml
from lazybridge.ext.report_builder.quarto.qmd import (
    render_fragment_to_qmd,
    render_report_to_qmd,
)


class TestQuartoDetect:
    def test_find_quarto_returns_path_or_none(self):
        # We don't assert presence — we assert the type contract.
        result = find_quarto()
        assert result is None or isinstance(result, str)

    def test_quarto_not_found_error_carries_install_hint(self):
        err = QuartoNotFoundError()
        assert "quarto.org" in str(err).lower() or "install" in str(err).lower()


class TestProjectYml:
    def test_html_section_present(self):
        yml = build_quarto_yml(formats=["html"], theme="cosmo", title="t")
        assert "html:" in yml
        assert "theme: cosmo" in yml

    def test_pdf_uses_typst_engine(self):
        yml = build_quarto_yml(formats=["pdf"], theme="cosmo", title="t")
        assert "typst:" in yml

    def test_bibliography_line_emitted(self):
        yml = build_quarto_yml(formats=["html"], theme="cosmo", bibliography="refs.json")
        assert "bibliography: refs.json" in yml


class TestQmdRendering:
    def test_text_fragment_with_heading(self):
        f = Fragment(kind="text", heading="Sub", body_md="Body markdown")
        out = render_fragment_to_qmd(f)
        assert "### Sub" in out
        assert "Body markdown" in out

    def test_callout_uses_pandoc_div(self):
        f = Fragment(kind="callout", body_md="Be careful", callout_style="warning")
        out = render_fragment_to_qmd(f)
        assert "::: {.callout-warning}" in out
        assert ":::" in out

    def test_table_renders_pipe_format(self):
        f = Fragment(
            kind="table",
            table=TableSpec(headers=["A", "B"], rows=[["1", "2"]], caption="Cap"),
        )
        out = render_fragment_to_qmd(f)
        assert "| A | B |" in out
        assert "|---|---|" in out
        assert "| 1 | 2 |" in out
        assert ": Cap" in out

    def test_table_escapes_pipe_in_cells(self):
        f = Fragment(
            kind="table",
            table=TableSpec(headers=["A", "B"], rows=[["a|b", "c"]]),
        )
        out = render_fragment_to_qmd(f)
        # The literal pipe in 'a|b' must be escaped to keep the row valid.
        assert "a\\|b" in out

    def test_chart_vegalite_block(self):
        f = Fragment(
            kind="chart",
            chart=ChartSpec(engine="vega-lite", spec={"mark": "bar"}, title="T"),
        )
        out = render_fragment_to_qmd(f)
        assert "```{vegalite}" in out
        assert ": T" in out

    def test_full_report_emits_yaml_and_audit_and_sources(self):
        bus = FragmentBus("rep")
        bus.append(
            Fragment(
                kind="text",
                heading="Hi",
                body_md="x",
                section="a",
                citations=[Citation(key="k1", title="t1", year=2024)],
                provenance=Provenance(step_name="s"),
            )
        )
        report = bus.assemble(title="Title With \"Quote\"")
        qmd = render_report_to_qmd(report, bibliography_path="refs.json")
        assert qmd.startswith("---\n")
        assert 'title: "Title With \\"Quote\\""' in qmd
        assert "bibliography: refs.json" in qmd
        assert "## Audit Trail" in qmd
        assert "## Sources" in qmd
        # The fragment heading went under the section heading.
        assert "## a" in qmd or "### Hi" in qmd


@pytest.mark.skipif(shutil.which("quarto") is None, reason="quarto CLI not on PATH")
class TestQuartoRenderSmoke:
    """Real-execution smoke tests — only run when Quarto is installed."""

    def _seed(self) -> FragmentBus:
        bus = FragmentBus("quarto-smoke", assembler=BlackboardAssembler())
        bus.append(Fragment(kind="text", heading="Hi", body_md="Hello world.", section="intro"))
        bus.append(
            Fragment(
                kind="chart",
                section="data",
                chart=ChartSpec(
                    engine="vega-lite",
                    title="Bar",
                    spec={
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "mark": "bar",
                        "data": {"values": [{"a": "A", "b": 1}, {"a": "B", "b": 2}]},
                        "encoding": {
                            "x": {"field": "a", "type": "ordinal"},
                            "y": {"field": "b", "type": "quantitative"},
                        },
                    },
                ),
            )
        )
        return bus

    def test_render_html(self, tmp_path):
        bus = self._seed()
        report = bus.assemble(title="Smoke")
        produced = QuartoExporter().export(report, tmp_path, formats=["html"])
        assert produced["html"].exists()
        assert produced["html"].stat().st_size > 1000
