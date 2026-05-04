"""Vega-Lite chart adapter.

Renders a :class:`ChartSpec` with ``engine="vega-lite"`` using
``vl-convert-python`` — a pure-Rust binding (deno_runtime + resvg + svg2pdf)
that needs no Node, no Chrome, and no display server.  This makes it the
right default for headless LLM pipelines.

Embed shapes
------------

* HTML: a single ``<div id="vega-XXXX">`` plus a ``<script>`` block that
  imports vega/vega-lite/vega-embed from a CDN and calls ``embed(...)``.
  This makes charts interactive in a browser without bundling JS.
* SVG: vector output suitable for PDF / DOCX embedding via Pandoc.
* PNG: raster fallback (only used when an exporter explicitly requests it).
"""

from __future__ import annotations

import json
from uuid import uuid4

from lazybridge.ext.report_builder.charts import RenderedChart
from lazybridge.ext.report_builder.fragments import ChartSpec


def validate(spec: dict) -> list[str]:
    """Return a list of human-readable problems with ``spec``.

    We don't run the full Vega-Lite JSON Schema (would require the schema
    bundled in vl-convert and is overkill for catching obvious bugs).
    Instead we sanity-check the shape an LLM tends to get wrong: the
    top-level dict must have a ``mark`` or ``layer`` (or ``hconcat``/
    ``vconcat``) and either inline data or a data URL.
    """
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    has_mark = any(k in spec for k in ("mark", "layer", "hconcat", "vconcat", "facet", "repeat"))
    if not has_mark:
        problems.append("spec must include one of: mark, layer, hconcat, vconcat, facet, repeat")
    return problems


def _splice_data(spec: dict, data: list[dict] | None) -> dict:
    """Inline tabular data into the spec, replacing ``data.values`` if present."""
    if data is None:
        return spec
    spec = dict(spec)  # shallow copy at the top level only
    spec["data"] = {"values": data}
    return spec


def _embed_html(spec: dict) -> str:
    """Generate the interactive HTML embed: a div + a vega-embed script tag.

    The vega/vega-lite/vega-embed scripts are loaded once per page from the
    jsdelivr CDN; multiple charts on the same page share the same scripts
    (browsers de-duplicate identical ``src=`` URLs in the cache).  Each chart
    gets a unique ``div`` id so embeds don't collide.
    """
    div_id = f"vega-{uuid4().hex[:8]}"
    spec_json = json.dumps(spec, default=str)
    head = (
        '<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>'
        '<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>'
        '<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>'
    )
    body = (
        f'<div id="{div_id}" class="lazybridge-vega-chart"></div>'
        "<script>"
        f"vegaEmbed('#{div_id}', {spec_json}, {{actions: false, renderer: 'svg'}});"
        "</script>"
    )
    return head + body


def render(chart_spec: ChartSpec, *, raster: bool = False) -> RenderedChart:
    """Render the spec to HTML and (optionally) SVG/PNG."""
    from lazybridge.ext.report_builder._deps import require_vl_convert

    spec = _splice_data(chart_spec.spec, chart_spec.data)
    problems = validate(spec)
    if problems:
        raise ValueError("Invalid Vega-Lite spec: " + "; ".join(problems))

    html = _embed_html(spec)
    svg: str | None = None
    png_bytes: bytes | None = None

    # SVG / PNG paths are gated on the optional dep — vl-convert is what
    # bridges Python to Vega-Lite without a JS runtime.
    if raster:
        vl = require_vl_convert()
        # vl-convert exposes both ``vegalite_to_svg`` and ``vegalite_to_png``
        # on the top-level module.  SVG is preferred for PDF/DOCX (vector),
        # PNG only when an explicit raster is needed.
        try:
            svg = vl.vegalite_to_svg(spec)
        except Exception as exc:
            raise RuntimeError(f"vl_convert.vegalite_to_svg failed: {exc}") from exc
        # PNG is best-effort: not every spec rasterizes cleanly; we still
        # have the SVG as the lossless option, so we treat PNG failures as
        # non-fatal.  Exporters that want PNG check for ``png_bytes`` and
        # gracefully fall back to SVG when missing.
        try:
            png_bytes = vl.vegalite_to_png(spec, scale=2.0)
        except Exception:
            png_bytes = None

    return RenderedChart(engine="vega-lite", html=html, svg=svg, png_bytes=png_bytes)
