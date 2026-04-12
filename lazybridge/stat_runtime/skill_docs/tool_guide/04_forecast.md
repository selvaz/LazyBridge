# forecast_model Tool Reference

## What It Does

`forecast_model` generates a point forecast with confidence intervals from a previously fitted model. It takes a `run_id` from a successful `fit_model` call, re-fits the model from the stored specification to obtain a live model object, and then produces the forecast. The re-fit approach avoids persisting fragile pickle objects while still supporting forecast-after-fit.

This tool is for generating forecasts AFTER fitting. To generate a forecast at fit time, use the `forecast_steps` parameter on `fit_model` instead.

## Tool Signature

```python
forecast_model(
    run_id: str,            # Required. Run ID from a prior fit_model call.
    steps: int,             # Required. Number of forecast steps.
    ci_level: float = 0.95, # Optional. Confidence interval level (0-1).
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `run_id` | `str` | Yes | -- | The 16-character hex run ID returned by a prior `fit_model` call. The run must exist and have `status: "success"`. |
| `steps` | `int` | Yes | -- | Number of out-of-sample forecast steps to generate. Must be a positive integer. The meaning of "step" depends on the data frequency (1 step = 1 day for daily data, 1 month for monthly, etc.). |
| `ci_level` | `float` | No | `0.95` | Confidence interval coverage level, between 0 and 1 exclusive. `0.95` produces a 95% CI. `0.99` produces a 99% CI. |

## Prerequisites

Before calling `forecast_model`, you must have a successful `fit_model` run:

1. Call `fit_model(...)` and capture the returned dict.
2. Verify `status` is `"success"` in the return value.
3. Extract the `run_id` from the return value.
4. Pass that `run_id` to `forecast_model`.

```python
# Step 1: Fit
result = fit_model(family="arima", target_col="ret", dataset_name="equities",
                   params={"order": [1, 1, 1]})

# Step 2: Check status
if result.get("status") == "success":
    run_id = result["run_id"]

    # Step 3: Forecast
    forecast = forecast_model(run_id=run_id, steps=30, ci_level=0.95)
```

## Return Value: ForecastResult

On success, returns a `ForecastResult` serialized as a JSON dict:

```json
{
  "family": "arima",
  "steps": 30,
  "point_forecast": [0.0012, 0.0015, 0.0011, ...],
  "lower_ci": [-0.0234, -0.0245, -0.0250, ...],
  "upper_ci": [0.0258, 0.0275, 0.0272, ...],
  "ci_level": 0.95,
  "dates": [],
  "extra": {}
}
```

### ForecastResult Fields

| Field | Type | Description |
|---|---|---|
| `family` | `str` | The model family that produced this forecast (`"arima"`, `"garch"`, `"markov"`, `"ols"`) |
| `steps` | `int` | Number of forecast steps (echoes the input) |
| `point_forecast` | `list[float]` | Length-`steps` list of point forecasts. Index 0 = first step ahead, index `steps-1` = last step ahead. |
| `lower_ci` | `list[float]` | Length-`steps` list of lower confidence interval bounds |
| `upper_ci` | `list[float]` | Length-`steps` list of upper confidence interval bounds |
| `ci_level` | `float` | The confidence level used (echoes the input, e.g. `0.95`) |
| `dates` | `list[str]` | Forecast dates if available. Usually empty (dates are not auto-generated). |
| `extra` | `dict` | Family-specific forecast extras. Contents depend on the model family (see below). |

On error:

```json
{"error": true, "type": "ValueError", "message": "Run 'abc123' not found"}
```

## Forecast Behavior by Model Family

### ARIMA Forecast

**Method**: Direct forecast from the fitted SARIMAX result object via `result.get_forecast(steps=steps)`.

**How it works**: ARIMA forecasting uses the model's state-space representation to project the series forward. The forecast incorporates the autoregressive structure, differencing, and moving average terms. Confidence intervals widen with the forecast horizon, reflecting increasing uncertainty.

**CI computation**: Analytically computed from the state-space representation. The `alpha` parameter passed to `summary_frame` is `1 - ci_level`.

**Extra fields**: None (empty dict).

**Output shape**:
- `point_forecast`: Forecasted mean values at each step
- `lower_ci` / `upper_ci`: CI bounds from `mean_ci_lower` / `mean_ci_upper` in the statsmodels summary frame

**Example**:

```python
forecast_model(run_id="abc123def456gh78", steps=20, ci_level=0.95)
```

### GARCH Forecast

**Method**: Variance forecast from the fitted arch model via `result.forecast(horizon=steps)`.

**How it works**: GARCH forecasts project the conditional variance (and mean) forward. The point forecast is the mean model forecast. Confidence intervals are constructed from the forecasted volatility: `mean +/- z * sqrt(variance)` where `z` is the normal quantile for the requested CI level.

**CI computation**: Uses `scipy.stats.norm.ppf((1 + ci_level) / 2)` for the z-score. CI bounds are `mean_forecast +/- z * volatility_forecast`.

**Extra fields**:

| Key | Type | Description |
|---|---|---|
| `variance_forecast` | `list[float]` | Forecasted conditional variance at each step |
| `volatility_forecast` | `list[float]` | Forecasted conditional volatility (sqrt of variance) at each step |

**Example**:

```python
forecast_model(run_id="garch_run_id_here", steps=10, ci_level=0.99)
```

### Markov Switching Forecast

**Method**: Regime-weighted prediction using the transition matrix and regime-specific intercepts.

**How it works**:
1. Starts from the last smoothed regime probabilities (from the fitted model).
2. At each forecast step, multiplies the current regime probability vector by the transpose of the transition matrix to get the next-step regime probabilities.
3. The point forecast at each step is the dot product of regime probabilities and regime-specific intercepts (`const[0]`, `const[1]`, etc.).
4. Confidence intervals use the overall residual standard deviation scaled by `sqrt(horizon)`.

**CI computation**: Uses `scipy.stats.norm.ppf((1 + ci_level) / 2)` for the z-score. CI width at step `h` is `z * std(residuals) * sqrt(h)`. CIs grow with the square root of the forecast horizon.

**Extra fields**:

| Key | Type | Description |
|---|---|---|
| `regime_probabilities_forecast` | `list[list[float]]` | At each step, the predicted regime probability vector. Length = `steps`, each inner list has `k_regimes` elements. |

**Example**:

```python
forecast_model(run_id="markov_run_id_here", steps=12, ci_level=0.90)
```

### OLS Forecast

**Method**: Trend extrapolation using `result.get_prediction(future_X)` from the statsmodels OLS result.

**How it works**: For OLS models fitted WITHOUT user-supplied exogenous variables (trend-only models), the forecast extends the time index. A constant + time index matrix is constructed for the forecast horizon: `sm.add_constant(np.arange(n, n + steps))`. The `get_prediction` method produces point forecasts and CI bounds.

**CI computation**: Uses the statsmodels prediction interval framework. The CI bounds come from `obs_ci_lower` / `obs_ci_upper` in the prediction summary frame, which accounts for both parameter uncertainty and residual variance.

**Rejects exogenous models**: If the original `fit_model` call included `exog_cols`, `forecast_model` raises an error because future values of the exogenous variables are not available. The error message is:

```
"OLS forecast is not supported for models fitted with exogenous regressors
because future values of the regressors are required but not available.
Either re-fit without exogenous variables, or use query_data to build
your own prediction matrix."
```

**Extra fields**: None (empty dict).

**Example**:

```python
# This works -- trend-only OLS
fit_result = fit_model(family="ols", target_col="price", dataset_name="stocks")
forecast_model(run_id=fit_result["run_id"], steps=30)

# This FAILS -- OLS with exogenous variables
fit_result = fit_model(family="ols", target_col="ret", dataset_name="equities",
                       exog_cols=["market_ret", "smb"])
forecast_model(run_id=fit_result["run_id"], steps=10)
# Returns: {"error": true, "type": "ValueError", "message": "OLS forecast is not supported ..."}
```

## Auto-Generated Plots

`forecast_model` does NOT auto-generate plots. Plots are only auto-generated during `fit_model` when the `forecast_steps` parameter is set. If you need a forecast plot from a `forecast_model` call, you must use `fit_model` with `forecast_steps` instead.

To get the forecast plot from a `fit_model` run that included `forecast_steps`:

```python
get_plot(run_id="...", name="forecast")
# Returns: {"path": "/abs/path/to/forecast.png", "name": "forecast", "description": "20-step forecast with 95% CI"}
```

The forecast plot shows the last 100 observations of actuals (black line), the point forecast (blue line), and the CI bands (shaded blue region).

## Common Errors

| Error Type | Cause | Message Pattern |
|---|---|---|
| `ValueError` | Run ID not found in the MetaStore | `"Run 'X' not found"` |
| `ValueError` | Run exists but did not succeed | `"Run 'X' did not succeed (status=failed)"` |
| `ValueError` | OLS model was fitted with exogenous variables | `"OLS forecast is not supported for models fitted with exogenous regressors..."` |
| `ValueError` | Re-fit fails because underlying data is no longer available | `"Dataset 'X' is not registered"` |
| `ImportError` | Required library not installed | `"statsmodels is required..."` or `"arch is required..."` |

### Re-Fit Failure

Because `forecast_model` re-fits the model from the stored spec, it can fail if:
- The dataset has been deregistered since the original fit.
- The underlying data file has been moved or deleted.
- The data file contents have changed (different columns, fewer rows).

The stored spec is in `RunRecord.spec_json`, which records the `dataset_name`, `query_sql`, `target_col`, `exog_cols`, and all `params`. The re-fit uses these exact same parameters.

## Examples

### ARIMA Forecast

```python
# Fit an ARIMA(1,1,1)
fit_result = fit_model(
    family="arima", target_col="price",
    dataset_name="stocks", params={"order": [1, 1, 1]}
)

# Generate 20-step forecast with 95% CI
forecast = forecast_model(
    run_id=fit_result["run_id"], steps=20, ci_level=0.95
)
# forecast["point_forecast"]  -> [101.2, 101.5, 101.8, ...]
# forecast["lower_ci"]        -> [98.1, 97.5, 96.9, ...]
# forecast["upper_ci"]        -> [104.3, 105.5, 106.7, ...]
```

### GARCH Forecast

```python
# Fit a GARCH(1,1)
fit_result = fit_model(
    family="garch", target_col="ret",
    dataset_name="equities", params={"p": 1, "q": 1}
)

# 10-step variance forecast with 99% CI
forecast = forecast_model(
    run_id=fit_result["run_id"], steps=10, ci_level=0.99
)
# forecast["extra"]["variance_forecast"]    -> [0.0004, 0.00038, ...]
# forecast["extra"]["volatility_forecast"]  -> [0.020, 0.0195, ...]
```

### Markov Forecast

```python
# Fit a 2-regime Markov model
fit_result = fit_model(
    family="markov", target_col="ret",
    dataset_name="equities", params={"k_regimes": 2}
)

# 12-step regime-weighted forecast
forecast = forecast_model(
    run_id=fit_result["run_id"], steps=12, ci_level=0.90
)
# forecast["extra"]["regime_probabilities_forecast"]  -> [[0.7, 0.3], [0.65, 0.35], ...]
```

### OLS Trend Forecast

```python
# Fit a trend-only OLS (no exog_cols)
fit_result = fit_model(
    family="ols", target_col="price", dataset_name="stocks"
)

# 30-step trend extrapolation
forecast = forecast_model(
    run_id=fit_result["run_id"], steps=30, ci_level=0.95
)
```

### Generating Forecasts at Different CI Levels

```python
run_id = fit_result["run_id"]
forecast_90 = forecast_model(run_id=run_id, steps=20, ci_level=0.90)
forecast_95 = forecast_model(run_id=run_id, steps=20, ci_level=0.95)
forecast_99 = forecast_model(run_id=run_id, steps=20, ci_level=0.99)
```

Each call re-fits the model, so calling multiple times is computationally expensive. If you need multiple CI levels, consider using `fit_model` with `forecast_steps` for the primary forecast, then `forecast_model` for additional CI levels.
