# analyze() Tool Reference

## What It Does

`analyze()` is the primary analysis tool. It accepts a goal-oriented `mode` instead of requiring you to pick a model family. The runtime automatically selects the appropriate model, fits it, runs diagnostics, generates plots, and returns enriched results with interpretation, assumptions, and suggested next steps.

## Tool Signature

```python
analyze(
    dataset_name: str,                  # Required. Registered dataset name
    target_col: str,                    # Required. Target column name
    mode: str = "recommend",            # Analysis goal (see modes below)
    time_col: str | None = None,        # Auto-detected from metadata if not set
    forecast_steps: int | None = None,  # Auto-set for forecast/volatility modes
    group_col: str | None = None,       # Column to filter by (e.g. "symbol")
    group_value: str | None = None,     # Value to filter to (e.g. "SPY")
    params: dict | None = None,         # Expert override: model parameters
)
```

## Analysis Modes

| Mode | What happens | Model selected |
|---|---|---|
| `recommend` | Inspects data roles and characteristics, picks best analysis | Auto (GARCH for returns, ARIMA for time series, OLS otherwise) |
| `describe` | Data profiling, stationarity tests, distribution analysis | OLS (baseline) |
| `forecast` | Time-series forecasting with auto forecast_steps=20 | ARIMA |
| `volatility` | Volatility modeling with auto forecast_steps=20 | GARCH |
| `regime` | Regime detection (bull/bear, high/low volatility) | Markov Switching |

You can also pass an explicit family name (`ols`, `arima`, `garch`, `markov`) as the mode for backward compatibility.

## Auto-Detection Features

- **Time column**: If `time_col` is not provided, it is auto-detected from dataset metadata (`time_column` field).
- **Forecast steps**: For `forecast` and `volatility` modes, `forecast_steps` defaults to 20 if not set.
- **Model family**: `mode="recommend"` examines column roles and data to choose:
  - Return-like target + time column → GARCH (volatility)
  - Time column present, non-return target → ARIMA (forecast)
  - No time column → OLS (regression)

## Panel Data Filtering

For datasets with multiple entities (e.g., multiple stock symbols), use `group_col` and `group_value`:

```python
analyze(
    dataset_name="equities",
    target_col="ret",
    mode="volatility",
    group_col="symbol",
    group_value="SPY"
)
```

This generates a SQL filter internally — no need to write `query_sql`.

## Return Value

Returns an `AnalysisResult` dict:

```json
{
  "run_id": "a1b2c3d4e5f6g7h8",
  "status": "success",
  "engine": "garch",
  "dataset_name": "equities",
  "target_col": "ret",
  "mode": "volatility",
  "mode_rationale": "Mode 'volatility' maps to GARCH family.",
  "assumptions": [
    "GARCH model assumes volatility clustering in the return series.",
    "Target should be returns, not prices.",
    "Default GARCH(1,1) with normal distribution."
  ],
  "params": {"mu": 0.05, "omega": 0.01, "alpha[1]": 0.08, "beta[1]": 0.90},
  "metrics": {"aic": 1234.56, "bic": 1267.89, "log_likelihood": -612.28},
  "fit_summary": "... model summary text ...",
  "diagnostics": [...],
  "diagnostics_passed": 3,
  "diagnostics_failed": 0,
  "model_adequate": true,
  "forecast": {"point_forecast": [...], "lower_ci": [...], "upper_ci": [...]},
  "plots": [
    {"name": "residuals", "artifact_type": "plot", "path": "artifacts/.../plots/residuals.png"},
    {"name": "volatility", "artifact_type": "plot", "path": "artifacts/.../plots/volatility.png"},
    {"name": "forecast", "artifact_type": "plot", "path": "artifacts/.../plots/forecast.png"}
  ],
  "data_artifacts": [
    {"name": "residuals", "artifact_type": "data", "path": "artifacts/.../data/residuals.json"}
  ],
  "interpretation": [
    "Volatility persistence (alpha+beta) = 0.9800.",
    "High persistence. Half-life of volatility shocks: 34 periods.",
    "Diagnostics: 3/3 tests passed."
  ],
  "warnings": [],
  "next_steps": [
    "Try EGARCH or TARCH for asymmetric volatility comparison.",
    "Try dist='t' (Student-t) for fat-tailed returns."
  ],
  "duration_secs": 2.5,
  "error_message": null
}
```

## Key Result Fields

| Field | Purpose |
|---|---|
| `mode` | The analysis mode that was used |
| `mode_rationale` | Why this analysis/family was chosen (especially useful for `recommend` mode) |
| `assumptions` | Key assumptions the model makes about the data |
| `model_adequate` | `true` if all key diagnostics pass (Ljung-Box); `false` otherwise |
| `interpretation` | Plain-English explanation of results |
| `warnings` | Concerning findings (e.g., non-convergence, failed diagnostics) |
| `next_steps` | Suggested follow-up actions |
| `plots` | All generated plot artifacts with file paths |

## Expert Override

To use specific model parameters while still getting enriched output, pass `params`:

```python
analyze(
    dataset_name="equities",
    target_col="ret",
    mode="volatility",
    params={"p": 1, "q": 1, "vol": "EGARCH", "dist": "t"}
)
```

## Error Handling

The tool never raises. On failure:
- `status` is `"failed"`
- `error_message` contains the exception details
- `warnings` and `next_steps` suggest recovery actions
- All other fields have safe defaults (empty lists, False, etc.)

## When to Use analyze() vs fit_model()

| Use case | Tool |
|---|---|
| Standard analysis, let runtime choose | `analyze(mode="recommend")` |
| Specific goal (forecast, volatility, regime) | `analyze(mode="forecast")` |
| Custom model parameters | `analyze(mode="volatility", params={...})` |
| Raw RunRecord output needed | `fit_model(family="garch", ...)` |
| Programmatic pipeline, no interpretation needed | `fit_model(...)` |
| Specific ARIMA(2,1,3) order selection | `fit_model(family="arima", params={"order": [2,1,3]})` |
