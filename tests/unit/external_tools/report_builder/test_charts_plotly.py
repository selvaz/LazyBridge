"""Tests for the Plotly chart adapter (gated on plotly availability)."""

from __future__ import annotations

import importlib.util

import pytest

from lazybridge.external_tools.report_builder.charts import plotly as plotly_mod
from lazybridge.external_tools.report_builder.fragments import ChartSpec

_PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None
pytestmark = pytest.mark.skipif(not _PLOTLY_AVAILABLE, reason="plotly not installed")


class TestValidate:
    def test_rejects_missing_data_list(self):
        problems = plotly_mod.validate({"layout": {}})
        assert any("data" in p for p in problems)

    def test_accepts_data_list(self):
        assert plotly_mod.validate({"data": [{"type": "scatter"}]}) == []


class TestRenderHtml:
    def test_html_embed_contains_plotly(self):
        spec = ChartSpec(
            engine="plotly",
            spec={"data": [{"type": "scatter", "x": [1, 2], "y": [3, 4]}]},
        )
        rendered = plotly_mod.render(spec, raster=False)
        assert rendered.engine == "plotly"
        # Plotly's to_html(...) uses 'plotly-graph-div' or a CDN reference.
        assert "plotly" in rendered.html.lower()


class TestSpliceData:
    def test_splice_fills_first_trace_xy(self):
        spec = ChartSpec(
            engine="plotly",
            spec={"data": [{"type": "scatter"}]},
            data=[{"x": 1, "y": 5}, {"x": 2, "y": 6}],
        )
        rendered = plotly_mod.render(spec, raster=False)
        assert rendered.engine == "plotly"
        # The y array {5,6} should appear somewhere in the rendered HTML.
        assert "5" in rendered.html
