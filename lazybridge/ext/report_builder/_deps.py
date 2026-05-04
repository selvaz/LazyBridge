"""Optional dependency checker for report_builder.

``import lazybridge.ext.report_builder`` never triggers these imports —
they fire only when the tool is actually called.

Usage inside other report_builder modules::

    from lazybridge.ext.report_builder._deps import require_markdown, require_bleach
"""

from __future__ import annotations

_INSTALL_HINT = "pip install lazybridge[report]"


def require_markdown():
    """Return the ``markdown`` module or raise with install instructions."""
    try:
        import markdown

        return markdown
    except ImportError:
        raise ImportError(f"markdown is required for this feature. Run: {_INSTALL_HINT}") from None


def require_bleach():
    """Return the ``bleach`` module or raise with install instructions."""
    try:
        import bleach

        return bleach
    except ImportError:
        raise ImportError(f"bleach is required for this feature. Run: {_INSTALL_HINT}") from None


def is_available(module_name: str) -> bool:
    """Check if a module is importable without raising."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def require_jinja2():
    """Return the ``jinja2`` module or raise with install instructions."""
    try:
        import jinja2

        return jinja2
    except ImportError:
        raise ImportError(f"jinja2 is required for this feature. Run: {_INSTALL_HINT}") from None


def require_weasyprint():
    """Return the ``weasyprint`` module or raise with install instructions."""
    try:
        import weasyprint

        return weasyprint
    except ImportError:
        raise ImportError("weasyprint is required for PDF output. Run: pip install lazybridge[pdf]") from None


# ---------------------------------------------------------------------------
# Fragment-based reporting subsystem deps
# ---------------------------------------------------------------------------


def require_vl_convert():
    """Return the ``vl_convert`` module — pure-Rust Vega-Lite renderer."""
    try:
        import vl_convert

        return vl_convert
    except ImportError:
        raise ImportError(
            "vl-convert-python is required for Vega-Lite chart rendering. "
            "Run: pip install lazybridge[report-charts]"
        ) from None


def require_plotly():
    """Return the ``plotly`` module — Plotly chart engine."""
    try:
        import plotly  # noqa: F401
        import plotly.io as pio

        return pio
    except ImportError:
        raise ImportError(
            "plotly is required for Plotly chart rendering. "
            "Run: pip install lazybridge[report-charts]"
        ) from None


def require_habanero():
    """Return the ``habanero.Crossref`` constructor for citation enrichment."""
    try:
        from habanero import Crossref

        return Crossref
    except ImportError:
        raise ImportError(
            "habanero is required for Crossref citation enrichment. "
            "Run: pip install lazybridge[report-citations]"
        ) from None


def require_httpx():
    """Return the ``httpx`` module — OpenAlex client + general HTTP fallback."""
    try:
        import httpx

        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for OpenAlex citation lookup. "
            "Run: pip install lazybridge[report-citations]"
        ) from None


def require_pypandoc():
    """Return the ``pypandoc`` module — Pandoc shell-out wrapper for fallback DOCX."""
    try:
        import pypandoc

        return pypandoc
    except ImportError:
        raise ImportError(
            "pypandoc is required for DOCX output via the WeasyPrint fallback path. "
            "Run: pip install lazybridge[report-fallback]"
        ) from None


def require_python_docx():
    """Return the ``docx`` (python-docx) module for surgical DOCX post-processing."""
    try:
        import docx

        return docx
    except ImportError:
        raise ImportError(
            "python-docx is required for DOCX post-processing. "
            "Run: pip install lazybridge[report-fallback]"
        ) from None


REPORT_AVAILABLE: bool = all(is_available(m) for m in ("markdown", "bleach", "jinja2"))
PDF_AVAILABLE: bool = is_available("weasyprint")
VL_CONVERT_AVAILABLE: bool = is_available("vl_convert")
PLOTLY_AVAILABLE: bool = is_available("plotly")
HABANERO_AVAILABLE: bool = is_available("habanero")
HTTPX_AVAILABLE: bool = is_available("httpx")
PYPANDOC_AVAILABLE: bool = is_available("pypandoc")
PYTHON_DOCX_AVAILABLE: bool = is_available("docx")
