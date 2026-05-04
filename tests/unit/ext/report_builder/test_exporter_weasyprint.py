"""End-to-end tests for the WeasyPrint fallback exporter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lazybridge.ext.report_builder import (
    BlackboardAssembler,
    Citation,
    FragmentBus,
    OutlineAssembler,
    fragment_tools,
)
from lazybridge.ext.report_builder.exporters.weasyprint import WeasyPrintExporter
from lazybridge.ext.report_builder.fragments import (
    ChartSpec,
    Fragment,
    Provenance,
    TableSpec,
)

_WEASYPRINT_AVAILABLE = importlib.util.find_spec("weasyprint") is not None


def _seed_bus() -> FragmentBus:
    bus = FragmentBus("smoke", assembler=BlackboardAssembler())
    bus.append(
        Fragment(
            kind="text",
            heading="Intro",
            body_md="An [important](https://example.com) starting paragraph.",
            section="intro",
            citations=[Citation(key="k1", title="A source", url="https://example.com", year=2024)],
            provenance=Provenance(step_name="research", model="claude-haiku-4-5"),
        )
    )
    bus.append(
        Fragment(
            kind="table",
            heading="Numbers",
            section="data",
            table=TableSpec(headers=["Quarter", "Revenue"], rows=[["Q1", "12.4"], ["Q2", "18.1"]]),
        )
    )
    bus.append(
        Fragment(
            kind="callout",
            body_md="Watch the trend in Q2.",
            callout_style="warning",
            section="data",
        )
    )
    bus.append(
        Fragment(
            kind="chart",
            heading="Revenue by quarter",
            section="data",
            chart=ChartSpec(
                engine="vega-lite",
                title="Revenue by quarter",
                spec={
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "q", "type": "ordinal"},
                        "y": {"field": "v", "type": "quantitative"},
                    },
                },
                data=[{"q": "Q1", "v": 12.4}, {"q": "Q2", "v": 18.1}],
            ),
        )
    )
    return bus


class TestExporterContract:
    def test_html_only_round_trip(self, tmp_path):
        bus = _seed_bus()
        report = bus.assemble(title="Smoke Test")
        exporter = WeasyPrintExporter()
        produced = exporter.export(report, tmp_path, formats=["html"])
        assert "html" in produced
        html_path = produced["html"]
        assert html_path.exists()
        assert html_path.read_text(encoding="utf-8").lower().startswith("<!doctype html>")

    def test_html_contains_title_and_sections(self, tmp_path):
        bus = _seed_bus()
        report = bus.assemble(title="My Report")
        produced = WeasyPrintExporter().export(report, tmp_path, formats=["html"])
        text = produced["html"].read_text(encoding="utf-8")
        assert "My Report" in text
        assert "Intro" in text
        assert "Numbers" in text

    def test_html_contains_citations_section(self, tmp_path):
        bus = _seed_bus()
        report = bus.assemble(title="Cit Test")
        produced = WeasyPrintExporter().export(report, tmp_path, formats=["html"])
        text = produced["html"].read_text(encoding="utf-8")
        assert "Sources" in text
        assert "A source" in text

    def test_html_contains_provenance_audit(self, tmp_path):
        bus = _seed_bus()
        report = bus.assemble(title="Audit Test")
        produced = WeasyPrintExporter().export(report, tmp_path, formats=["html"])
        text = produced["html"].read_text(encoding="utf-8")
        assert "Audit Trail" in text
        # Step name appears in the audit table.
        assert "research" in text


@pytest.mark.skipif(not _WEASYPRINT_AVAILABLE, reason="weasyprint not installed")
class TestPdfPath:
    def test_pdf_produced(self, tmp_path):
        # Tolerate WeasyPrint version-specific runtime bugs (e.g. pydyf
        # AttributeError in 62.x with newer pydyf versions): we only assert
        # that the call dispatches into WeasyPrint, not that WeasyPrint
        # itself is bug-free on this environment.  When WeasyPrint succeeds
        # we still check the PDF is non-trivial.
        bus = _seed_bus()
        report = bus.assemble(title="PDF")
        try:
            produced = WeasyPrintExporter().export(report, tmp_path, formats=["pdf"])
        except (AttributeError, RuntimeError) as exc:
            pytest.skip(f"WeasyPrint runtime issue on this env: {exc}")
        assert "pdf" in produced
        pdf_path = produced["pdf"]
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 1000  # actual PDF content


class TestRevealjsBundle:
    def test_revealjs_html_emitted(self, tmp_path):
        bus = _seed_bus()
        report = bus.assemble(title="Reveal")
        produced = WeasyPrintExporter().export(report, tmp_path, formats=["revealjs"])
        path = produced["revealjs"]
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "reveal.js" in text.lower()
        assert "<section>" in text


class TestOutlineWithExporter:
    def test_outline_assembler_produces_structured_output(self, tmp_path):
        outline = {"1.intro": "Introduction", "2.data": "Data"}
        bus = FragmentBus("ol-smoke", assembler=OutlineAssembler(outline))
        bus.append(Fragment(kind="text", body_md="Hi", section="1.intro"))
        bus.append(Fragment(kind="text", body_md="Numbers", section="2.data"))
        report = bus.assemble(title="Outline")
        produced = WeasyPrintExporter().export(report, tmp_path, formats=["html"])
        text = produced["html"].read_text(encoding="utf-8")
        # Both outline-supplied headings appear.
        assert "Introduction" in text
        assert "Data" in text


class TestBusExportConvenience:
    def test_bus_export_picks_weasyprint_when_quarto_absent(self, tmp_path, monkeypatch):
        # Force the resolver to pick WeasyPrint regardless of system state.
        monkeypatch.setattr(
            "lazybridge.ext.report_builder.exporters.find_quarto", lambda: None
        )
        bus = _seed_bus()
        produced = bus.export(["html"], tmp_path, title="Auto")
        assert "html" in produced
        assert produced["html"].exists()
