"""OLS (Ordinary Least Squares) engine via statsmodels."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from lazybridge.ext.stat_runtime.engines.base import BaseEngine
from lazybridge.ext.stat_runtime.schemas import (
    DiagnosticResult,
    FitResult,
    ForecastResult,
    ModelFamily,
    ModelSpec,
)


class OLSEngine(BaseEngine):
    family: ClassVar[ModelFamily] = ModelFamily.OLS

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        *,
        spec: ModelSpec,
    ) -> FitResult:
        sm = _import_sm()

        has_exog = X is not None
        if X is None:
            X = sm.add_constant(np.arange(len(y), dtype=float))
        elif spec.params.get("add_constant", True):
            X = sm.add_constant(X)

        model = sm.OLS(y, X)
        result = model.fit()

        params = {name: float(val) for name, val in zip(result.model.exog_names, result.params)}
        metrics = {
            "r_squared": float(result.rsquared),
            "adj_r_squared": float(result.rsquared_adj),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "log_likelihood": float(result.llf),
            "f_statistic": float(result.fvalue) if result.fvalue is not None else 0.0,
            "f_pvalue": float(result.f_pvalue) if result.f_pvalue is not None else 1.0,
        }
        p_values = {name: float(pv) for name, pv in zip(result.model.exog_names, result.pvalues)}

        return FitResult(
            family=ModelFamily.OLS,
            summary_text=str(result.summary()),
            params=params,
            metrics=metrics,
            residuals_json=result.resid.tolist(),
            fitted_values_json=result.fittedvalues.tolist(),
            extra={
                "has_exog": has_exog,
                "p_values": p_values,
                "std_errors": {name: float(se) for name, se in zip(result.model.exog_names, result.bse)},
                "_result_obj": result,  # kept in memory for forecast, not persisted
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
            raise ValueError("OLS forecast requires a fitted model object (re-fit first)")

        # Check if the model was fitted with user-supplied exogenous variables.
        # If so, we cannot generate a meaningful forecast without future X values.
        has_exog = fit_result.extra.get("has_exog", False)
        if has_exog:
            raise ValueError(
                "OLS forecast is not supported for models fitted with exogenous "
                "regressors because future values of the regressors are required "
                "but not available. Either re-fit without exogenous variables, or "
                "use query_data to build your own prediction matrix."
            )

        sm = _import_sm()
        n = len(fit_result.fitted_values_json)
        future_X = sm.add_constant(np.arange(n, n + steps, dtype=float))
        predictions = result_obj.get_prediction(future_X)
        summary = predictions.summary_frame(alpha=1 - ci_level)

        return ForecastResult(
            family=ModelFamily.OLS,
            steps=steps,
            point_forecast=summary["mean"].tolist(),
            lower_ci=summary["obs_ci_lower"].tolist(),
            upper_ci=summary["obs_ci_upper"].tolist(),
            ci_level=ci_level,
        )

    def diagnostics(
        self,
        fit_result: FitResult,
        y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> list[DiagnosticResult]:
        from lazybridge.ext.stat_runtime.diagnostics import (
            durbin_watson_test,
            jarque_bera_test,
            ljung_box_test,
        )

        residuals = np.array(fit_result.residuals_json)
        return [
            durbin_watson_test(residuals),
            jarque_bera_test(residuals),
            ljung_box_test(residuals),
        ]


def _import_sm():
    from lazybridge.ext.stat_runtime._deps import require_statsmodels

    require_statsmodels()
    import statsmodels.api as sm

    return sm
