"""Markov Switching Regression engine via statsmodels."""

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


class MarkovEngine(BaseEngine):
    family: ClassVar[ModelFamily] = ModelFamily.MARKOV

    def fit(
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,
        *,
        spec: ModelSpec,
    ) -> FitResult:
        MarkovRegression = _import_markov()

        k_regimes = spec.params.get("k_regimes", 2)
        order = spec.params.get("order", 0)
        trend = spec.params.get("trend", "c")
        switching_variance = spec.params.get("switching_variance", True)

        model = MarkovRegression(
            y,
            k_regimes=k_regimes,
            order=order,
            trend=trend,
            exog=X,
            switching_variance=switching_variance,
        )
        result = model.fit(disp=False)

        params = {name: float(val) for name, val in zip(result.param_names, result.params)}

        # Extract transition matrix
        transition_matrix = result.regime_transition.tolist()

        # Smoothed regime probabilities
        smoothed_probs = {}
        for regime in range(k_regimes):
            smoothed_probs[f"regime_{regime}"] = result.smoothed_marginal_probabilities[regime].tolist()

        metrics = {
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
            "log_likelihood": float(result.llf),
        }

        # Regime means and variances
        regime_info = {}
        for regime in range(k_regimes):
            regime_info[f"regime_{regime}_duration"] = float(
                1.0 / (1.0 - result.regime_transition[regime, regime])
            ) if result.regime_transition[regime, regime] < 1.0 else float("inf")

        return FitResult(
            family=ModelFamily.MARKOV,
            summary_text=str(result.summary()),
            params=params,
            metrics=metrics,
            residuals_json=result.resid.tolist(),
            fitted_values_json=result.fittedvalues.tolist() if hasattr(result, "fittedvalues") else [],
            extra={
                "k_regimes": k_regimes,
                "transition_matrix": transition_matrix,
                "smoothed_probabilities": smoothed_probs,
                "regime_info": regime_info,
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
            raise ValueError("Markov forecast requires a fitted model object (re-fit first)")

        # Markov models have limited forecast support — use regime-weighted prediction
        k_regimes = fit_result.extra.get("k_regimes", 2)
        transition = np.array(fit_result.extra["transition_matrix"])

        # Get last regime probabilities
        last_probs = np.array([
            fit_result.extra["smoothed_probabilities"][f"regime_{r}"][-1]
            for r in range(k_regimes)
        ])

        # Extract regime-specific intercepts from params
        regime_means = []
        for r in range(k_regimes):
            key = f"const[{r}]"
            regime_means.append(fit_result.params.get(key, 0.0))
        regime_means = np.array(regime_means)

        point_forecast = []
        regime_probs_forecast = []
        current_probs = last_probs.copy()
        for _ in range(steps):
            current_probs = transition.T @ current_probs
            point_forecast.append(float(current_probs @ regime_means))
            regime_probs_forecast.append(current_probs.tolist())

        # CI grows with forecast horizon (sqrt scaling)
        residuals = np.array(fit_result.residuals_json)
        std = float(np.std(residuals)) if len(residuals) > 0 else 1.0
        from scipy.stats import norm
        z = norm.ppf((1 + ci_level) / 2)

        pf = np.array(point_forecast)
        horizons = np.arange(1, steps + 1)
        ci_width = z * std * np.sqrt(horizons)
        return ForecastResult(
            family=ModelFamily.MARKOV,
            steps=steps,
            point_forecast=pf.tolist(),
            lower_ci=(pf - ci_width).tolist(),
            upper_ci=(pf + ci_width).tolist(),
            ci_level=ci_level,
            extra={"regime_probabilities_forecast": regime_probs_forecast},
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
        residuals = np.array(fit_result.residuals_json)
        results = [
            ljung_box_test(residuals, lags=min(10, len(residuals) // 5)),
            jarque_bera_test(residuals),
        ]

        # Regime classification quality
        k_regimes = fit_result.extra.get("k_regimes", 2)
        smoothed = fit_result.extra.get("smoothed_probabilities", {})
        if smoothed:
            max_probs = np.max(
                [smoothed.get(f"regime_{r}", [0]) for r in range(k_regimes)],
                axis=0,
            )
            avg_certainty = float(np.mean(max_probs))
            results.append(DiagnosticResult(
                test_name="Regime Classification Certainty",
                statistic=avg_certainty,
                passed=avg_certainty > 0.7,
                interpretation=(
                    f"Average regime classification certainty: {avg_certainty:.3f}. "
                    f"{'Good' if avg_certainty > 0.7 else 'Weak'} regime separation "
                    f"(threshold: 0.70)."
                ),
            ))

        return results


def _import_markov():
    from lazybridge.ext.stat_runtime._deps import require_statsmodels
    require_statsmodels()
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    return MarkovRegression
