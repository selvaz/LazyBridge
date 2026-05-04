"""Optional dependency checker for stat_runtime.

Every heavy dependency (duckdb, polars, sqlglot, statsmodels, arch, matplotlib)
is lazy-imported.  ``import lazybridge.external_tools.stat_runtime`` never triggers these
imports — they fire only when a feature that needs them is actually called.

Usage inside other stat_runtime modules::

    from lazybridge.external_tools.stat_runtime._deps import require_duckdb

    def some_function():
        duckdb = require_duckdb()
        conn = duckdb.connect()
        ...
"""

from __future__ import annotations

_INSTALL_HINT = "pip install lazybridge[stats]"


def require_duckdb():
    """Return the ``duckdb`` module or raise with install instructions."""
    try:
        import duckdb

        return duckdb
    except ImportError:
        raise ImportError(f"duckdb is required for this feature. Run: {_INSTALL_HINT}") from None


def require_polars():
    """Return the ``polars`` module or raise with install instructions."""
    try:
        import polars

        return polars
    except ImportError:
        raise ImportError(f"polars is required for this feature. Run: {_INSTALL_HINT}") from None


def require_sqlglot():
    """Return the ``sqlglot`` module or raise with install instructions."""
    try:
        import sqlglot

        return sqlglot
    except ImportError:
        raise ImportError(f"sqlglot is required for this feature. Run: {_INSTALL_HINT}") from None


def require_statsmodels():
    """Return the ``statsmodels`` top-level package or raise."""
    try:
        import statsmodels

        return statsmodels
    except ImportError:
        raise ImportError(f"statsmodels is required for this feature. Run: {_INSTALL_HINT}") from None


def require_arch():
    """Return the ``arch`` module or raise with install instructions."""
    try:
        import arch

        return arch
    except ImportError:
        raise ImportError(f"arch is required for this feature. Run: {_INSTALL_HINT}") from None


def require_matplotlib():
    """Return ``matplotlib.pyplot`` or raise with install instructions."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend for headless use
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(f"matplotlib is required for this feature. Run: {_INSTALL_HINT}") from None


def is_available(module_name: str) -> bool:
    """Check if a module is importable without raising."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


STATS_AVAILABLE: bool = all(is_available(m) for m in ("duckdb", "polars", "sqlglot", "statsmodels", "arch"))
