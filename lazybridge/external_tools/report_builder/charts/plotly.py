"""Plotly chart adapter.

Renders a :class:`ChartSpec` with ``engine="plotly"`` using the official
``plotly`` Python library.  Interactive HTML is produced via
``plotly.io.to_html`` with the Plotly.js bundle pulled from the CDN — a
single ``<script>`` tag is shared across charts on the same page.

Raster rendering is best-effort and gated behind ``kaleido``: kaleido v1
(2025) requires a system Chrome on ``$PATH`` which is a sharp edge for
container deployments.  When kaleido fails or isn't installed, the
exporter that requested raster falls back to embedding the interactive
HTML — slightly degraded for PDF / DOCX but never broken.
"""

from __future__ import annotations

from lazybridge.external_tools.report_builder.charts import RenderedChart
from lazybridge.external_tools.report_builder.fragments import ChartSpec


def validate(spec: dict) -> list[str]:
    """Sanity-check a Plotly figure dict.

    A valid Plotly figure is a dict with ``data`` (list) and optional
    ``layout``.  We don't full-schema-check; just make sure the LLM didn't
    hand us a Vega-Lite spec by mistake.
    """
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    if "data" not in spec or not isinstance(spec["data"], list):
        problems.append("Plotly spec must include 'data': list[trace]")
    return problems


def _splice_data(spec: dict, data: list[dict] | None) -> dict:
    """Fold inline data rows into the first trace's x/y arrays.

    LLMs find it easier to emit ``data=[{x:1,y:2}, {x:3,y:4}]`` than to
    interleave the values into trace arrays.  When data is supplied AND
    the first trace is missing x/y, we splice them in by column.  This is
    a deliberately narrow convenience — for more complex charts the LLM
    should populate the spec directly.
    """
    if data is None:
        return spec
    spec = {**spec, "data": list(spec.get("data", []))}
    if not spec["data"]:
        # Synthesise a minimal scatter trace if the LLM forgot.
        spec["data"] = [{"type": "scatter", "mode": "markers"}]
    trace = dict(spec["data"][0])
    if data and isinstance(data[0], dict):
        keys = list(data[0].keys())
        if "x" not in trace and len(keys) >= 1:
            trace["x"] = [row.get(keys[0]) for row in data]
        if "y" not in trace and len(keys) >= 2:
            trace["y"] = [row.get(keys[1]) for row in data]
    spec["data"][0] = trace
    return spec


def render(chart_spec: ChartSpec, *, raster: bool = False) -> RenderedChart:
    from lazybridge.external_tools.report_builder._deps import require_plotly

    pio = require_plotly()
    spec = _splice_data(chart_spec.spec, chart_spec.data)
    problems = validate(spec)
    if problems:
        raise ValueError("Invalid Plotly spec: " + "; ".join(problems))

    # ``pio.from_json`` accepts a JSON string; we already have a dict so we
    # use the lower-level Figure constructor directly via ``plotly.graph_objects``.
    import plotly.graph_objects as go

    fig = go.Figure(spec)
    # ``include_plotlyjs="cdn"`` keeps the page weight low and lets the
    # browser cache plotly.js across reports.  ``full_html=False`` returns a
    # standalone snippet we can drop into Quarto/Jinja output.
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

    svg: str | None = None
    png_bytes: bytes | None = None
    if raster:
        # Best-effort raster.  kaleido import + render can fail if Chrome
        # isn't on the system; we swallow that and let the exporter fall
        # back to the HTML embed.
        try:
            png_bytes = pio.to_image(fig, format="png", scale=2.0)
        except Exception:
            png_bytes = None
        try:
            svg_bytes = pio.to_image(fig, format="svg")
            svg = svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
        except Exception:
            svg = None

    return RenderedChart(engine="plotly", html=html, svg=svg, png_bytes=png_bytes)
