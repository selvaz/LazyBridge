"""Unit tests for the Fragment / Citation / Provenance schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lazybridge.ext.report_builder.fragments import (
    ChartSpec,
    Citation,
    Fragment,
    Provenance,
    TableSpec,
)


class TestFragmentValidation:
    def test_text_requires_body_md(self):
        with pytest.raises(ValueError, match="non-empty body_md"):
            Fragment(kind="text", body_md="")

    def test_text_with_only_whitespace_body_rejected(self):
        with pytest.raises(ValueError, match="non-empty body_md"):
            Fragment(kind="text", body_md="   ")

    def test_chart_requires_chart_field(self):
        with pytest.raises(ValueError, match="requires a 'chart' field"):
            Fragment(kind="chart")

    def test_table_requires_table_field(self):
        with pytest.raises(ValueError, match="requires a 'table' field"):
            Fragment(kind="table")

    def test_callout_defaults_style_to_note(self):
        f = Fragment(kind="callout", body_md="hi")
        assert f.callout_style == "note"

    def test_text_round_trip(self):
        f = Fragment(kind="text", heading="H", body_md="Body", section="intro", order_hint=1.5)
        d = f.model_dump(mode="json")
        f2 = Fragment.model_validate(d)
        assert f2.heading == "H"
        assert f2.body_md == "Body"
        assert f2.section == "intro"
        assert f2.order_hint == 1.5
        assert f2.id == f.id

    def test_unique_ids_by_default(self):
        a = Fragment(kind="text", body_md="x")
        b = Fragment(kind="text", body_md="x")
        assert a.id != b.id

    def test_chart_fragment(self):
        chart = ChartSpec(engine="vega-lite", spec={"mark": "bar"})
        f = Fragment(kind="chart", chart=chart)
        assert f.chart is chart

    def test_table_fragment(self):
        t = TableSpec(headers=["A", "B"], rows=[["1", "2"]])
        f = Fragment(kind="table", table=t)
        assert f.table.headers == ["A", "B"]


class TestCitation:
    def test_to_csl_json_minimal(self):
        c = Citation(key="smith2024", title="A paper", year=2024, authors=["Smith, A."])
        csl = c.to_csl_json()
        assert csl["id"] == "smith2024"
        assert csl["title"] == "A paper"
        assert csl["author"] == [{"literal": "Smith, A."}]
        assert csl["issued"] == {"date-parts": [[2024]]}

    def test_to_csl_json_with_url_and_doi(self):
        c = Citation(
            key="ref",
            title="t",
            url="https://example.com",
            doi="10.1234/abc",
        )
        csl = c.to_csl_json()
        assert csl["URL"] == "https://example.com"
        assert csl["DOI"] == "10.1234/abc"

    def test_to_csl_json_uses_pre_enriched_csl_when_present(self):
        prebuilt = {"id": "old", "type": "book", "title": "Different"}
        c = Citation(key="newkey", title="Whatever", csl=prebuilt)
        csl = c.to_csl_json()
        assert csl["id"] == "newkey"  # we override id with our key
        assert csl["title"] == "Different"  # but trust the rest of the prebuilt record


class TestProvenance:
    def test_timestamp_default_is_utc(self):
        p = Provenance()
        assert p.timestamp.tzinfo is not None
        # utcoffset of UTC is timedelta(0)
        assert p.timestamp.utcoffset().total_seconds() == 0

    def test_round_trip(self):
        p = Provenance(
            step_name="research",
            agent_name="claude",
            model="claude-haiku-4-5",
            tokens_in=120,
            tokens_out=400,
            cost_usd=0.012,
            latency_ms=842.0,
        )
        d = p.model_dump(mode="json")
        p2 = Provenance.model_validate(d)
        assert p2.step_name == "research"
        assert p2.tokens_in == 120
        assert p2.cost_usd == 0.012


class TestChartSpec:
    def test_default_engine_is_vega_lite(self):
        spec = ChartSpec(spec={"mark": "bar"})
        assert spec.engine == "vega-lite"

    def test_data_optional(self):
        spec = ChartSpec(spec={"mark": "bar"})
        assert spec.data is None
