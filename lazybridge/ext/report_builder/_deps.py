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


REPORT_AVAILABLE: bool = all(is_available(m) for m in ("markdown", "bleach", "jinja2"))
PDF_AVAILABLE: bool = is_available("weasyprint")
