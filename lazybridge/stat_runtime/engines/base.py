"""Abstract base class for model engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from lazybridge.stat_runtime.schemas import (
    DiagnosticResult,
    FitResult,
    ForecastResult,
    ModelFamily,
    ModelSpec,
)


class BaseEngine(ABC):
    """Abstract model engine.  One subclass per ModelFamily.

    Engines receive numpy arrays, not DataFrames.  The runner is responsible
    for extracting columns from the query result before calling the engine.
    """

    family: ClassVar[ModelFamily]

    @abstractmethod
    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        *,
        spec: ModelSpec,
    ) -> FitResult:
        """Fit the model and return structured results."""
        ...

    @abstractmethod
    def forecast(
        self,
        fit_result: FitResult,
        steps: int,
        *,
        ci_level: float = 0.95,
    ) -> ForecastResult:
        """Generate forecasts from a fitted model."""
        ...

    @abstractmethod
    def diagnostics(
        self,
        fit_result: FitResult,
        y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> list[DiagnosticResult]:
        """Run model-specific diagnostic tests."""
        ...
