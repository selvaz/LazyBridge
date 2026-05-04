"""Chart engine adapters for the fragment-based report builder.

Two engines ship by default:

* :mod:`.vega`   — Vega-Lite v5 specs rendered via ``vl-convert-python``
                   (pure-Rust, no Node, no Chrome, no display server).
                   Default for LLM-emitted charts.
* :mod:`.plotly` — Plotly figure-spec dicts rendered to interactive HTML
                   via ``plotly.io.to_html`` and to raster (best-effort)
                   via ``kaleido``.  Opt-in.

Both adapters return a :class:`RenderedChart` with the embeddable HTML and
optional raster bytes for non-HTML formats.  An invalid spec returns a
structured error dict from the validator helpers — never raises into the
LLM tool path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderedChart:
    """The render product of a :class:`ChartSpec`.

    ``html`` is a self-contained ``<div>``-based snippet (or ``<script>``
    block) embeddable in any HTML document.  ``svg`` / ``png_bytes`` are
    populated lazily for non-HTML exporters.
    """

    engine: str
    html: str
    svg: str | None = None
    png_bytes: bytes | None = None


def render_chart(chart_spec, *, raster: bool = False) -> RenderedChart:
    """Dispatch to the right engine based on ``ChartSpec.engine``."""
    from lazybridge.ext.report_builder.fragments import ChartSpec

    assert isinstance(chart_spec, ChartSpec)
    if chart_spec.engine == "vega-lite":
        from lazybridge.ext.report_builder.charts import vega

        return vega.render(chart_spec, raster=raster)
    if chart_spec.engine == "plotly":
        from lazybridge.ext.report_builder.charts import plotly as _plotly

        return _plotly.render(chart_spec, raster=raster)
    raise ValueError(f"Unknown chart engine: {chart_spec.engine!r}")


__all__ = ["RenderedChart", "render_chart"]
