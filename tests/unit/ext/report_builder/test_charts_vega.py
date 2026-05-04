"""Tests for the Vega-Lite chart adapter."""

from __future__ import annotations

import importlib.util

import pytest

from lazybridge.ext.report_builder.charts import vega
from lazybridge.ext.report_builder.fragments import ChartSpec

_VL_CONVERT_AVAILABLE = importlib.util.find_spec("vl_convert") is not None


class TestValidate:
    def test_rejects_non_dict(self):
        problems = vega.validate("not a dict")  # type: ignore[arg-type]
        assert problems

    def test_rejects_missing_mark(self):
        problems = vega.validate({"foo": "bar"})
        assert any("mark" in p for p in problems)

    def test_accepts_mark(self):
        assert vega.validate({"mark": "bar"}) == []

    def test_accepts_layer(self):
        assert vega.validate({"layer": []}) == []


class TestRenderHtml:
    def test_html_embed_contains_div_and_script(self):
        spec = ChartSpec(spec={"mark": "bar", "encoding": {}})
        rendered = vega.render(spec, raster=False)
        assert rendered.engine == "vega-lite"
        assert "<div" in rendered.html
        assert "vegaEmbed" in rendered.html
        # CDN script tags present once.
        assert rendered.html.count("vega-embed@6") >= 1

    def test_render_with_inline_data_splices_values(self):
        spec = ChartSpec(
            spec={"mark": "bar"},
            data=[{"x": 1, "y": 10}, {"x": 2, "y": 20}],
        )
        rendered = vega.render(spec, raster=False)
        # Inline data values should appear in the spec dump.
        assert '"values"' in rendered.html
        assert "10" in rendered.html

    def test_unique_ids_across_renders(self):
        # Two renders produce distinct div ids so they don't collide on the
        # same page.
        spec = ChartSpec(spec={"mark": "bar"})
        a = vega.render(spec, raster=False)
        b = vega.render(spec, raster=False)
        # Extract the id attribute crudely; they shouldn't match.
        import re

        ids_a = re.findall(r'id="(vega-[a-f0-9]+)"', a.html)
        ids_b = re.findall(r'id="(vega-[a-f0-9]+)"', b.html)
        assert ids_a and ids_b
        assert ids_a[0] != ids_b[0]


@pytest.mark.skipif(not _VL_CONVERT_AVAILABLE, reason="vl-convert-python not installed")
class TestRasterizationGated:
    def test_svg_produced(self):
        spec = ChartSpec(
            spec={
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": "bar",
                "data": {"values": [{"a": "A", "b": 1}, {"a": "B", "b": 2}]},
                "encoding": {
                    "x": {"field": "a", "type": "ordinal"},
                    "y": {"field": "b", "type": "quantitative"},
                },
            }
        )
        rendered = vega.render(spec, raster=True)
        assert rendered.svg is not None
        assert rendered.svg.startswith("<svg")
