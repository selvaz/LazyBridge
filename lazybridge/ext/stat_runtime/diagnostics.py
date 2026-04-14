"""Stationarity tests, residual diagnostics, and model comparison.

Every function returns a DiagnosticResult with a human-readable interpretation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lazybridge.ext.stat_runtime.schemas import DiagnosticResult, RunRecord

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stationarity tests
# ---------------------------------------------------------------------------

def adf_test(
    series: np.ndarray,
    *,
    significance: float = 0.05,
    regression: str = "c",
) -> DiagnosticResult:
    """Augmented Dickey-Fuller test for unit roots."""
    sm_tsa = _import_sm_tsa()
    result = sm_tsa.stattools.adfuller(series, regression=regression)
    stat, pvalue = float(result[0]), float(result[1])
    passed = pvalue < significance

    return DiagnosticResult(
        test_name="Augmented Dickey-Fuller",
        statistic=stat,
        p_value=pvalue,
        passed=passed,
        detail={
            "used_lag": result[2],
            "n_obs": result[3],
            "critical_values": {k: float(v) for k, v in result[4].items()},
        },
        interpretation=(
            f"ADF statistic = {stat:.4f}, p-value = {pvalue:.4f}. "
            f"{'Series is stationary' if passed else 'Series is non-stationary (has unit root)'} "
            f"at the {significance*100:.0f}% significance level."
        ),
    )


def kpss_test(
    series: np.ndarray,
    *,
    significance: float = 0.05,
    regression: str = "c",
    nlags: str = "auto",
) -> DiagnosticResult:
    """KPSS test for stationarity (null = stationary)."""
    sm_tsa = _import_sm_tsa()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, lags, crit = sm_tsa.stattools.kpss(
            series, regression=regression, nlags=nlags,
        )
    stat, pvalue = float(stat), float(pvalue)
    # KPSS: reject null (stationary) if p < significance → NOT stationary
    passed = pvalue >= significance  # fail to reject → stationary

    return DiagnosticResult(
        test_name="KPSS",
        statistic=stat,
        p_value=pvalue,
        passed=passed,
        detail={
            "lags_used": lags,
            "critical_values": {k: float(v) for k, v in crit.items()},
        },
        interpretation=(
            f"KPSS statistic = {stat:.4f}, p-value = {pvalue:.4f}. "
            f"{'Series is stationary' if passed else 'Series is NOT stationary'} "
            f"at the {significance*100:.0f}% level. "
            f"(KPSS null hypothesis: series is stationary.)"
        ),
    )


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------

def ljung_box_test(
    residuals: np.ndarray,
    lags: int = 10,
    *,
    significance: float = 0.05,
) -> DiagnosticResult:
    """Ljung-Box test for serial correlation in residuals."""
    sm_diag = _import_sm_diagnostic()
    lags = max(1, min(lags, len(residuals) // 2 - 1))
    result = sm_diag.acorr_ljungbox(residuals, lags=[lags], return_df=True)
    stat = float(result["lb_stat"].iloc[0])
    pvalue = float(result["lb_pvalue"].iloc[0])
    passed = pvalue > significance

    return DiagnosticResult(
        test_name="Ljung-Box",
        statistic=stat,
        p_value=pvalue,
        passed=passed,
        detail={"lags": lags},
        interpretation=(
            f"Ljung-Box Q({lags}) = {stat:.4f}, p-value = {pvalue:.4f}. "
            f"{'No significant serial correlation' if passed else 'Serial correlation detected'} "
            f"in residuals at the {significance*100:.0f}% level."
        ),
    )


def jarque_bera_test(
    residuals: np.ndarray,
    *,
    significance: float = 0.05,
) -> DiagnosticResult:
    """Jarque-Bera test for normality of residuals."""
    sm_stats = _import_sm_stats()
    stat, pvalue, skew, kurtosis = sm_stats.jarque_bera(residuals)
    stat, pvalue = float(stat), float(pvalue)
    passed = pvalue > significance

    return DiagnosticResult(
        test_name="Jarque-Bera",
        statistic=stat,
        p_value=pvalue,
        passed=passed,
        detail={"skewness": float(skew), "kurtosis": float(kurtosis)},
        interpretation=(
            f"Jarque-Bera = {stat:.4f}, p-value = {pvalue:.4f}. "
            f"Skew = {skew:.4f}, Kurtosis = {kurtosis:.4f}. "
            f"{'Residuals are approximately normal' if passed else 'Residuals deviate from normality'} "
            f"at the {significance*100:.0f}% level."
        ),
    )


def durbin_watson_test(residuals: np.ndarray) -> DiagnosticResult:
    """Durbin-Watson statistic for first-order autocorrelation."""
    sm_stats = _import_sm_stats()
    stat = float(sm_stats.durbin_watson(residuals))

    if stat < 1.5:
        interp = "positive autocorrelation"
        passed = False
    elif stat > 2.5:
        interp = "negative autocorrelation"
        passed = False
    else:
        interp = "no significant first-order autocorrelation"
        passed = True

    return DiagnosticResult(
        test_name="Durbin-Watson",
        statistic=stat,
        passed=passed,
        detail={"range": "0-4, ideal ~2"},
        interpretation=(
            f"Durbin-Watson = {stat:.4f}. "
            f"Indicates {interp} (range: 0-4, ideal near 2)."
        ),
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(runs: list[RunRecord]) -> DiagnosticResult:
    """Compare multiple model runs by AIC and BIC."""
    if not runs:
        return DiagnosticResult(
            test_name="Model Comparison",
            interpretation="No runs provided for comparison.",
        )

    comparison: list[dict[str, Any]] = []
    for run in runs:
        comparison.append({
            "run_id": run.run_id,
            "engine": run.engine,
            "dataset": run.dataset_name,
            "aic": run.metrics_json.get("aic"),
            "bic": run.metrics_json.get("bic"),
            "log_likelihood": run.metrics_json.get("log_likelihood"),
        })

    # Find best by AIC and BIC
    valid_aic = [c for c in comparison if c["aic"] is not None]
    valid_bic = [c for c in comparison if c["bic"] is not None]
    best_aic = min(valid_aic, key=lambda c: c["aic"]) if valid_aic else None
    best_bic = min(valid_bic, key=lambda c: c["bic"]) if valid_bic else None

    interp_parts = [f"Compared {len(runs)} model runs."]
    if best_aic:
        interp_parts.append(f"Best AIC: {best_aic['engine']} (run {best_aic['run_id'][:8]}, AIC={best_aic['aic']:.2f}).")
    if best_bic:
        interp_parts.append(f"Best BIC: {best_bic['engine']} (run {best_bic['run_id'][:8]}, BIC={best_bic['bic']:.2f}).")

    return DiagnosticResult(
        test_name="Model Comparison",
        detail={"models": comparison},
        interpretation=" ".join(interp_parts),
    )


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _import_sm_tsa():
    from lazybridge.ext.stat_runtime._deps import require_statsmodels
    require_statsmodels()
    import statsmodels.tsa.stattools as _stattools
    # Return a namespace object that exposes .stattools
    class _ns:
        stattools = _stattools
    return _ns()


def _import_sm_diagnostic():
    from lazybridge.ext.stat_runtime._deps import require_statsmodels
    require_statsmodels()
    from statsmodels.stats import diagnostic
    return diagnostic


def _import_sm_stats():
    from lazybridge.ext.stat_runtime._deps import require_statsmodels
    require_statsmodels()
    from statsmodels.stats import stattools
    return stattools
