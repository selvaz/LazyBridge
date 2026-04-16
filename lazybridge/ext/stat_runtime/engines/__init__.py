"""Model engine registry.  Lazy registration — engines only imported on demand."""

from __future__ import annotations

from lazybridge.ext.stat_runtime.engines.base import BaseEngine
from lazybridge.ext.stat_runtime.schemas import ModelFamily

# Registry populated lazily on first get_engine() call per family
_REGISTRY: dict[ModelFamily, type[BaseEngine]] = {}
_LOADED: set[ModelFamily] = set()


def get_engine(family: ModelFamily | str) -> BaseEngine:
    """Return an engine instance for the given model family.

    Engines are lazy-imported to avoid pulling in statsmodels/arch at import time.
    """
    family = ModelFamily(family)

    if family not in _LOADED:
        _load_engine(family)

    cls = _REGISTRY.get(family)
    if cls is None:
        raise ValueError(f"No engine registered for family '{family}'")
    return cls()


def _load_engine(family: ModelFamily) -> None:
    """Import the engine module for a given family (lazy registration)."""
    if family == ModelFamily.OLS:
        from lazybridge.ext.stat_runtime.engines.ols import OLSEngine

        _REGISTRY[ModelFamily.OLS] = OLSEngine
    elif family == ModelFamily.ARIMA:
        from lazybridge.ext.stat_runtime.engines.arima import ARIMAEngine

        _REGISTRY[ModelFamily.ARIMA] = ARIMAEngine
    elif family == ModelFamily.GARCH:
        from lazybridge.ext.stat_runtime.engines.garch import GARCHEngine

        _REGISTRY[ModelFamily.GARCH] = GARCHEngine
    elif family == ModelFamily.MARKOV:
        from lazybridge.ext.stat_runtime.engines.markov import MarkovEngine

        _REGISTRY[ModelFamily.MARKOV] = MarkovEngine
    _LOADED.add(family)


__all__ = ["BaseEngine", "get_engine"]
