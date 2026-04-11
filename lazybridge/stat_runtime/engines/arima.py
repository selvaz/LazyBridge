"""ARIMA / SARIMAX engine via statsmodels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from lazybridge.stat_runtime.engines.base import BaseEngine
from lazybridge.stat_runtime.schemas import (
    DiagnosticResult,
    FitResult,
    ForecastResult,
    ModelFamily,
    ModelSpec,
)


class ARIMAEngine(BaseEngine):
    family: ClassVar[ModelFamily] = ModelFamily.ARIMA

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        *,
        spec: ModelSpec,
    ) -> FitResult:
        SARIMAX = _import_sarimax()

        order = tuple(spec.params.get("order", (1, 0, 0)))
        seasonal_order = tuple(spec.params.get("seasonal_order", (0, 0, 0, 0)))
        trend = spec.params.get("trend", "c")

        model = SARIMAX(
            y,
            exog=X,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=spec.params.get("enforce_stationarity", False),
            enforce_invertibility=spec.params.get("enforce_invertibility", False),
        )
        result = model.fit(disp=False)

        params = {name: float(val) for name, val in zip(result.param_names, result.params)}
        metrics = {
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
            "log_likelihood": float(result.llf),
        }

        return FitResult(
            family=ModelFamily.ARIMA,
            summary_text=str(result.summary()),
            params=params,
            metrics=metrics,
            residuals_json=result.resid.tolist(),
            fitted_values_json=result.fittedvalues.tolist(),
            extra={
                "order": list(order),
                "seasonal_order": list(seasonal_order),
                "_result_obj": result,
            },
        )

    def forecast(
        self,
        fit_result: FitResult,
        steps: int,
        *,
        ci_level: float = 0.95,
    ) -> ForecastResult:
        result_obj = fit_result.extra.get("_result_obj")
        if result_obj is None:
            raise ValueError("ARIMA forecast requires a fitted model object (re-fit first)")

        alpha = 1 - ci_level
        forecast = result_obj.get_forecast(steps=steps)
        summary = forecast.summary_frame(alpha=alpha)

        return ForecastResult(
            family=ModelFamily.ARIMA,
            steps=steps,
            point_forecast=summary["mean"].tolist(),
            lower_ci=summary[f"mean_ci_lower"].tolist(),
            upper_ci=summary[f"mean_ci_upper"].tolist(),
            ci_level=ci_level,
        )

    def diagnostics(
        self,
        fit_result: FitResult,
        y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> list[DiagnosticResult]:
        from lazybridge.stat_runtime.diagnostics import (
            jarque_bera_test,
            ljung_box_test,
        )
        residuals = np.array(fit_result.residuals_json)
        return [
            ljung_box_test(residuals, lags=min(10, len(residuals) // 5)),
            jarque_bera_test(residuals),
        ]


def _import_sarimax():
    from lazybridge.stat_runtime._deps import require_statsmodels
    require_statsmodels()
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    return SARIMAX
