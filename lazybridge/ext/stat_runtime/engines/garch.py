"""GARCH-family engine via the arch library."""

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


class GARCHEngine(BaseEngine):
    family: ClassVar[ModelFamily] = ModelFamily.GARCH

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        *,
        spec: ModelSpec,
    ) -> FitResult:
        arch_model = _import_arch_model()

        p = spec.params.get("p", 1)
        q = spec.params.get("q", 1)
        vol = spec.params.get("vol", "GARCH")
        dist = spec.params.get("dist", "normal")
        mean = spec.params.get("mean", "Constant")
        rescale = spec.params.get("rescale", False)

        model = arch_model(
            y, mean=mean, vol=vol, p=p, q=q, dist=dist, rescale=rescale,
        )
        result = model.fit(disp="off")

        params = {name: float(val) for name, val in result.params.items()}
        cond_vol = result.conditional_volatility

        metrics = {
            "aic": float(result.aic),
            "bic": float(result.bic),
            "log_likelihood": float(result.loglikelihood),
        }

        # Standardized residuals: guard against division by zero
        safe_vol = np.where(cond_vol > 0, cond_vol, np.nan)
        std_resid = np.where(safe_vol > 0, result.resid / safe_vol, 0.0)

        return FitResult(
            family=ModelFamily.GARCH,
            summary_text=str(result.summary()),
            params=params,
            metrics=metrics,
            residuals_json=result.resid.tolist(),
            fitted_values_json=result.resid.tolist(),  # mean model residuals (not volatility)
            extra={
                "p": p,
                "q": q,
                "vol_model": vol,
                "distribution": dist,
                "conditional_volatility": cond_vol.tolist(),
                "std_residuals": std_resid.tolist(),
                "p_values": {name: float(pv) for name, pv in result.pvalues.items()},
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
            raise ValueError("GARCH forecast requires a fitted model object (re-fit first)")

        from scipy.stats import norm
        z = norm.ppf((1 + ci_level) / 2)

        forecasts = result_obj.forecast(horizon=steps)
        mean_fc = forecasts.mean.iloc[-1].values
        var_fc = forecasts.variance.iloc[-1].values
        vol_fc = np.sqrt(var_fc)

        return ForecastResult(
            family=ModelFamily.GARCH,
            steps=steps,
            point_forecast=mean_fc.tolist(),
            lower_ci=(mean_fc - z * vol_fc).tolist(),
            upper_ci=(mean_fc + z * vol_fc).tolist(),
            ci_level=ci_level,
            extra={
                "variance_forecast": var_fc.tolist(),
                "volatility_forecast": vol_fc.tolist(),
            },
        )

    def diagnostics(
        self,
        fit_result: FitResult,
        y: np.ndarray,
        X: np.ndarray | None = None,
    ) -> list[DiagnosticResult]:
        from lazybridge.ext.stat_runtime.diagnostics import (
            jarque_bera_test,
            ljung_box_test,
        )
        std_resid = np.array(fit_result.extra.get("std_residuals", fit_result.residuals_json))
        results = [
            ljung_box_test(std_resid, lags=10),
            ljung_box_test(std_resid ** 2, lags=10),
            jarque_bera_test(std_resid),
        ]
        # Rename the squared residuals test
        results[1].test_name = "Ljung-Box (squared residuals)"
        results[1].interpretation = (
            results[1].interpretation.replace("residuals", "squared standardized residuals")
        )
        return results


def _import_arch_model():
    from lazybridge.ext.stat_runtime._deps import require_arch
    require_arch()
    from arch import arch_model
    return arch_model
