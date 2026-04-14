"""Visualization module — publication-ready statistical plots.

All plots use matplotlib with a consistent style. Each function saves the
figure to the artifact store and returns the file path.

Plot catalog:
  - plot_residuals     — residual scatter + histogram
  - plot_acf_pacf      — ACF and PACF side-by-side
  - plot_volatility    — GARCH conditional volatility
  - plot_regimes       — Markov smoothed regime probabilities
  - plot_forecast      — forecast with confidence bands
  - plot_model_comparison — AIC/BIC bar chart
  - plot_series        — basic time series plot
"""

from __future__ import annotations

import logging

import numpy as np

from lazybridge.stat_runtime.artifact_store import ArtifactStore
from lazybridge.stat_runtime.schemas import ForecastResult, RunRecord

_logger = logging.getLogger(__name__)

# Consistent style
_STYLE = {
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}


def _get_plt():
    from lazybridge.stat_runtime._deps import require_matplotlib
    plt = require_matplotlib()
    plt.rcParams.update(_STYLE)
    return plt


# ---------------------------------------------------------------------------
# Residual plots
# ---------------------------------------------------------------------------

def plot_residuals(
    residuals: np.ndarray | list[float],
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    title: str = "Residual Analysis",
) -> str:
    """Residual scatter plot + histogram.  Returns artifact path."""
    plt = _get_plt()
    residuals = np.asarray(residuals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter
    ax1.plot(residuals, "o", markersize=2, alpha=0.6)
    ax1.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax1.set_title("Residuals over Time")
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("Residual")

    # Histogram
    ax2.hist(residuals, bins=min(50, len(residuals) // 5 + 1), edgecolor="black", alpha=0.7)
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, "residuals", fig,
        description="Residual scatter plot and histogram",
    )


# ---------------------------------------------------------------------------
# ACF / PACF
# ---------------------------------------------------------------------------

def plot_acf_pacf(
    series: np.ndarray | list[float],
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    lags: int = 40,
    title: str = "ACF & PACF",
) -> str:
    """Autocorrelation and partial autocorrelation plots."""
    plt = _get_plt()
    from lazybridge.stat_runtime._deps import require_statsmodels
    require_statsmodels()
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    series = np.asarray(series)
    lags = min(lags, len(series) // 2 - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(series, lags=lags, ax=ax1, alpha=0.05)
    plot_pacf(series, lags=lags, ax=ax2, alpha=0.05, method="ywm")
    ax1.set_title("Autocorrelation (ACF)")
    ax2.set_title("Partial Autocorrelation (PACF)")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, "acf_pacf", fig,
        description="ACF and PACF correlogram",
    )


# ---------------------------------------------------------------------------
# Conditional volatility (GARCH)
# ---------------------------------------------------------------------------

def plot_volatility(
    conditional_vol: np.ndarray | list[float],
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    returns: np.ndarray | list[float] | None = None,
    title: str = "Conditional Volatility (GARCH)",
) -> str:
    """Plot GARCH conditional volatility, optionally with returns overlay."""
    plt = _get_plt()
    vol = np.asarray(conditional_vol)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if returns is not None:
        ret = np.asarray(returns)
        ax1.plot(ret, color="gray", alpha=0.5, linewidth=0.8, label="Returns")
        ax1.set_ylabel("Returns", color="gray")
        ax2 = ax1.twinx()
        ax2.plot(vol, color="crimson", linewidth=1.2, label="Conditional Volatility")
        ax2.set_ylabel("Volatility", color="crimson")
        ax2.legend(loc="upper right")
        ax1.legend(loc="upper left")
    else:
        ax1.plot(vol, color="crimson", linewidth=1.2)
        ax1.set_ylabel("Conditional Volatility")
        ax1.fill_between(range(len(vol)), 0, vol, alpha=0.15, color="crimson")

    ax1.set_xlabel("Observation")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, "volatility", fig,
        description="GARCH conditional volatility plot",
    )


# ---------------------------------------------------------------------------
# Regime probabilities (Markov)
# ---------------------------------------------------------------------------

def plot_regimes(
    smoothed_probs: dict[str, list[float]],
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    series: np.ndarray | list[float] | None = None,
    title: str = "Regime Probabilities (Markov Switching)",
) -> str:
    """Plot smoothed regime probabilities with optional data overlay."""
    plt = _get_plt()
    n_regimes = len(smoothed_probs)

    fig, axes = plt.subplots(n_regimes + (1 if series is not None else 0), 1,
                             figsize=(12, 3 * (n_regimes + (1 if series is not None else 0))),
                             sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800"]
    idx = 0

    if series is not None:
        axes[idx].plot(np.asarray(series), color="black", linewidth=0.8)
        axes[idx].set_ylabel("Data")
        axes[idx].set_title("Observed Series")
        idx += 1

    for i, (regime_name, probs) in enumerate(smoothed_probs.items()):
        color = colors[i % len(colors)]
        axes[idx].fill_between(range(len(probs)), 0, probs, alpha=0.4, color=color)
        axes[idx].plot(probs, color=color, linewidth=1)
        axes[idx].set_ylabel("Probability")
        axes[idx].set_title(f"P({regime_name.replace('_', ' ').title()})")
        axes[idx].set_ylim(0, 1)
        idx += 1

    axes[-1].set_xlabel("Observation")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, "regimes", fig,
        description="Markov switching smoothed regime probabilities",
    )


# ---------------------------------------------------------------------------
# Forecast plot
# ---------------------------------------------------------------------------

def plot_forecast(
    actuals: np.ndarray | list[float],
    forecast_result: ForecastResult,
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    last_n: int = 100,
    title: str = "Forecast",
) -> str:
    """Plot actuals with forecast and confidence interval bands."""
    plt = _get_plt()
    actuals = np.asarray(actuals)
    n_actual = len(actuals)
    show_from = max(0, n_actual - last_n)
    shown_actuals = actuals[show_from:]

    fc = np.array(forecast_result.point_forecast)
    lo = np.array(forecast_result.lower_ci)
    hi = np.array(forecast_result.upper_ci)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Actuals
    x_actual = range(show_from, n_actual)
    ax.plot(x_actual, shown_actuals, color="black", linewidth=1, label="Actual")

    # Forecast
    x_fc = range(n_actual, n_actual + len(fc))
    ax.plot(x_fc, fc, color="#2196F3", linewidth=1.5, label="Forecast")
    ax.fill_between(x_fc, lo, hi, alpha=0.2, color="#2196F3",
                     label=f"{forecast_result.ci_level*100:.0f}% CI")

    ax.set_xlabel("Observation")
    ax.set_ylabel("Value")
    ax.legend()
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, "forecast", fig,
        description=f"{forecast_result.steps}-step forecast with {forecast_result.ci_level*100:.0f}% CI",
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    runs: list[RunRecord],
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    title: str = "Model Comparison",
) -> str:
    """Bar chart comparing AIC and BIC across model runs."""
    plt = _get_plt()

    labels = [f"{r.engine}\n({r.run_id[:6]})" for r in runs]
    aics = [r.metrics_json.get("aic", 0) for r in runs]
    bics = [r.metrics_json.get("bic", 0) for r in runs]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))
    bars1 = ax.bar(x - width / 2, aics, width, label="AIC", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x + width / 2, bics, width, label="BIC", color="#FF5722", alpha=0.8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Information Criterion")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, "model_comparison", fig,
        description="AIC/BIC comparison across model runs",
    )


# ---------------------------------------------------------------------------
# Basic time series plot
# ---------------------------------------------------------------------------

def plot_series(
    series: np.ndarray | list[float],
    run_id: str,
    artifact_store: ArtifactStore,
    *,
    title: str = "Time Series",
    ylabel: str = "Value",
    name: str = "series",
) -> str:
    """Simple time series line plot."""
    plt = _get_plt()
    series = np.asarray(series)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(series, linewidth=0.8)
    ax.set_xlabel("Observation")
    ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    return artifact_store.write_plot(
        run_id, name, fig,
        description=f"Time series plot: {title}",
    )
