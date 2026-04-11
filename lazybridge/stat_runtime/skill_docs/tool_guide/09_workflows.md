# End-to-End Workflows

This document provides complete, step-by-step workflows using the stat_runtime tools. Each workflow shows every tool call in sequence with expected outputs and decision points.

---

## Workflow A: GARCH Volatility Study

**Goal**: Estimate time-varying volatility of a financial return series, assess model adequacy, and forecast future volatility.

### Step A1: Register the Data

```python
register_dataset(
    name="sp500",
    uri="/data/sp500_returns.parquet",
    time_column="date",
    frequency="daily"
)
```

Verify the return contains `"row_count"` and `"schema_json"` with the expected columns.

### Step A2: Profile the Data

```python
profile_dataset(name="sp500")
```

Check for:
- `null_pct` on the return column -- if > 5%, data quality may be an issue.
- `mean` and `std` of the return column -- returns should be small numbers centered near 0.
- If returns are in decimal form (e.g. mean ~ 0.0005), consider setting `rescale=true` in the GARCH params or pre-multiplying by 100 via a SQL query.

### Step A3: Run Stationarity Tests

```python
run_diagnostics(series_name="sp500", column="ret")
```

**Decision logic**:
- If ADF `passed=true` AND KPSS `passed=true`: series is stationary. Proceed to GARCH fitting.
- If ADF `passed=false` (unit root detected): the return series may need differencing. This is unusual for returns -- check if you are accidentally using price levels instead of returns.
- If KPSS `passed=false` (non-stationary): consider differencing or detrending.

For returns data, both tests should typically pass.

### Step A4: Fit GARCH(1,1)

```python
result = fit_model(
    family="garch",
    target_col="ret",
    dataset_name="sp500",
    params={"p": 1, "q": 1, "dist": "normal"},
    forecast_steps=20
)
```

Record the `run_id` from the return value. Check:
- `status` == `"success"`
- `metrics_json.aic` -- note this value for comparison
- `params_json`: verify `alpha[1] + beta[1] < 1.0` for stationarity of the variance process
- `diagnostics_json`: the Ljung-Box test on squared residuals should pass (no remaining ARCH effects)

### Step A5: Try Alternative Specifications

Fit GARCH(1,1) with Student-t errors for fat tails:

```python
result_t = fit_model(
    family="garch",
    target_col="ret",
    dataset_name="sp500",
    params={"p": 1, "q": 1, "dist": "t"},
    forecast_steps=20
)
```

Fit EGARCH for asymmetric volatility (leverage effect):

```python
result_egarch = fit_model(
    family="garch",
    target_col="ret",
    dataset_name="sp500",
    params={"p": 1, "q": 1, "vol": "EGARCH", "dist": "t"},
    forecast_steps=20
)
```

### Step A6: Compare Models

```python
compare_models(run_ids=[
    result["run_id"],
    result_t["run_id"],
    result_egarch["run_id"]
])
```

The `interpretation` field will identify the model with the lowest AIC and BIC. Choose the model with the best (lowest) information criterion.

### Step A7: Retrieve Volatility Plot

```python
best_run_id = result_t["run_id"]  # or whichever won
get_plot(run_id=best_run_id, name="volatility")
```

The returned path points to a PNG showing conditional volatility overlaid on returns.

### Step A8: Retrieve Forecast

```python
get_plot(run_id=best_run_id, name="forecast")
```

For the forecast data (not just the plot):

```python
list_artifacts(run_id=best_run_id, artifact_type="forecast")
```

The forecast artifact JSON contains `point_forecast`, `lower_ci`, `upper_ci`, `variance_forecast`, and `volatility_forecast`.

---

## Workflow B: ARIMA Forecasting

**Goal**: Fit an ARIMA model to a time series, validate the specification, and produce a multi-step forecast.

### Step B1: Register and Explore Data

```python
register_dataset(
    name="monthly_sales",
    uri="/data/sales.parquet",
    time_column="month",
    frequency="monthly"
)

profile_dataset(name="monthly_sales")
```

### Step B2: Inspect the Series with SQL

```python
query_data(
    sql="SELECT month, sales FROM dataset('monthly_sales') ORDER BY month LIMIT 10"
)
```

Check that the data is sorted by time and has reasonable values.

### Step B3: Test Stationarity

```python
run_diagnostics(series_name="monthly_sales", column="sales")
```

**Decision logic**:
- If non-stationary (ADF fails): use `d=1` in the ARIMA order.
- If still non-stationary after first difference: use `d=2`. (Rare -- usually `d=1` suffices.)
- If stationary: use `d=0`.

For seasonal data, also consider seasonal differencing via `seasonal_order`.

### Step B4: Fit Initial ARIMA

Start with ARIMA(1,1,1):

```python
result_111 = fit_model(
    family="arima",
    target_col="sales",
    dataset_name="monthly_sales",
    params={"order": [1, 1, 1], "trend": "c"},
    forecast_steps=12
)
```

### Step B5: Check Residual Diagnostics

From the return value, examine `diagnostics_json`:

- **Ljung-Box**: if `passed=false`, residuals have autocorrelation -- the model is under-specified. Increase `p` or `q`.
- **Jarque-Bera**: if `passed=false`, residuals are non-normal. This is informative but less critical for point forecasting.

Also examine the ACF/PACF plot of residuals:

```python
get_plot(run_id=result_111["run_id"], name="acf_pacf")
```

If significant spikes remain in the ACF at lag k, consider increasing `q` to k. If spikes remain in the PACF at lag k, consider increasing `p` to k.

### Step B6: Try Alternative Orders

```python
result_211 = fit_model(
    family="arima",
    target_col="sales",
    dataset_name="monthly_sales",
    params={"order": [2, 1, 1]},
    forecast_steps=12
)

result_112 = fit_model(
    family="arima",
    target_col="sales",
    dataset_name="monthly_sales",
    params={"order": [1, 1, 2]},
    forecast_steps=12
)
```

### Step B7: Try Seasonal ARIMA

If the data has monthly seasonality:

```python
result_seasonal = fit_model(
    family="arima",
    target_col="sales",
    dataset_name="monthly_sales",
    params={
        "order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 12]
    },
    forecast_steps=12
)
```

### Step B8: Compare and Select

```python
compare_models(run_ids=[
    result_111["run_id"],
    result_211["run_id"],
    result_112["run_id"],
    result_seasonal["run_id"]
])
```

Select the model with lowest AIC/BIC that also passes Ljung-Box on residuals.

### Step B9: Extended Forecast

If you need a longer forecast from the best model:

```python
forecast_model(
    run_id=result_seasonal["run_id"],
    steps=24,
    ci_level=0.95
)
```

Note: `forecast_model` requires the fitted model object to be in memory (same session as `fit_model`). If the session has ended, re-fit the model.

### Step B10: Retrieve Results

```python
get_plot(run_id=result_seasonal["run_id"], name="forecast")
list_artifacts(run_id=result_seasonal["run_id"])
```

---

## Workflow C: Markov Regime Detection

**Goal**: Identify distinct market regimes (e.g. bull/bear, high-vol/low-vol) in a financial time series.

### Step C1: Prepare Data

```python
register_dataset(
    name="market",
    uri="/data/market_returns.parquet",
    time_column="date",
    frequency="daily"
)

profile_dataset(name="market")
```

### Step C2: Fit 2-Regime Model

```python
result_2reg = fit_model(
    family="markov",
    target_col="ret",
    dataset_name="market",
    params={
        "k_regimes": 2,
        "switching_variance": true,
        "trend": "c"
    }
)
```

### Step C3: Interpret Regime Parameters

From `result_2reg["params_json"]`:
- `const[0]` and `const[1]`: regime-specific mean returns. The regime with higher mean is the "bull" or "calm" regime; lower mean is "bear" or "turbulent".
- `sigma2[0]` and `sigma2[1]` (if `switching_variance=true`): regime-specific variances. Higher variance = more volatile regime.

From `result_2reg["extra"]` (accessed via `get_run`):
- `transition_matrix`: check diagonal values. Values > 0.95 mean regimes are persistent.
- `regime_info`: expected duration in each regime.

### Step C4: Check Regime Separation Quality

In `diagnostics_json`, look for the "Regime Classification Certainty" test:
- `statistic > 0.70` and `passed=true`: good regime separation.
- `statistic < 0.70`: regimes are not well-separated. Consider different specification or fewer regimes.

### Step C5: Visualize Regimes

```python
get_plot(run_id=result_2reg["run_id"], name="regimes")
```

This produces a multi-panel plot with the observed series on top and smoothed regime probabilities below. Clear regime separation shows probabilities oscillating between 0 and 1 with sharp transitions.

### Step C6: Try 3-Regime Model

```python
result_3reg = fit_model(
    family="markov",
    target_col="ret",
    dataset_name="market",
    params={
        "k_regimes": 3,
        "switching_variance": true
    }
)
```

### Step C7: Compare 2 vs 3 Regimes

```python
compare_models(run_ids=[result_2reg["run_id"], result_3reg["run_id"]])
```

**Decision logic**:
- If the 3-regime model has substantially lower AIC/BIC, it captures additional structure.
- If AIC/BIC are similar, prefer the simpler 2-regime model.
- Also check the Regime Classification Certainty for both -- more regimes often leads to weaker separation.

### Step C8: Add AR Dynamics

If residuals show autocorrelation:

```python
result_ar = fit_model(
    family="markov",
    target_col="ret",
    dataset_name="market",
    params={
        "k_regimes": 2,
        "order": 1,
        "switching_variance": true
    }
)
```

### Step C9: Forecast Regime Evolution

```python
fc = forecast_model(
    run_id=result_2reg["run_id"],
    steps=30,
    ci_level=0.90
)
```

The forecast `extra` field contains `regime_probabilities_forecast` -- predicted regime probabilities at each future step.

---

## Workflow D: Model Comparison Across Families

**Goal**: Determine the best model family and specification for a given dataset by systematically comparing OLS, ARIMA, GARCH, and Markov models.

### Step D1: Register Data

```python
register_dataset(
    name="returns",
    uri="/data/returns.parquet",
    time_column="date",
    frequency="daily"
)
```

### Step D2: Baseline -- OLS Trend

```python
result_ols = fit_model(
    family="ols",
    target_col="ret",
    dataset_name="returns",
    forecast_steps=20
)
```

This fits a simple time trend. Useful as a naive baseline.

### Step D3: ARIMA

```python
result_arima = fit_model(
    family="arima",
    target_col="ret",
    dataset_name="returns",
    params={"order": [1, 0, 1]},
    forecast_steps=20
)
```

### Step D4: GARCH

```python
result_garch = fit_model(
    family="garch",
    target_col="ret",
    dataset_name="returns",
    params={"p": 1, "q": 1, "dist": "t"},
    forecast_steps=20
)
```

### Step D5: Markov Switching

```python
result_markov = fit_model(
    family="markov",
    target_col="ret",
    dataset_name="returns",
    params={"k_regimes": 2, "switching_variance": true},
    forecast_steps=20
)
```

### Step D6: Compare All Models

```python
comparison = compare_models(run_ids=[
    result_ols["run_id"],
    result_arima["run_id"],
    result_garch["run_id"],
    result_markov["run_id"]
])
```

### Step D7: Analyze Comparison Results

The `comparison["detail"]["models"]` list contains per-model metrics. The `comparison["interpretation"]` identifies the best model by AIC and BIC.

**Important caveats when comparing across families**:
- AIC/BIC values are directly comparable only when models are fitted on the same data (same observations, same target variable).
- OLS and ARIMA model the conditional mean; GARCH models conditional variance; Markov models regime-switching behavior. They answer different questions.
- A GARCH model with lower AIC than ARIMA does not mean GARCH is "better at forecasting returns" -- it means GARCH better captures the data distribution (including heteroskedasticity).

### Step D8: Review Diagnostics for Best Model

```python
best_id = result_garch["run_id"]  # based on comparison
run = get_run(run_id=best_id)
```

Check `diagnostics_json` for all diagnostic tests. A good model should pass:
- Ljung-Box on residuals (no serial correlation)
- Ljung-Box on squared residuals (no remaining ARCH effects, for GARCH)
- Jarque-Bera (residual normality, less critical)

### Step D9: Retrieve All Plots

```python
# Get all plots for the best model
plots = list_artifacts(run_id=best_id, artifact_type="plot")

# Individual retrieval
get_plot(run_id=best_id, name="residuals")
get_plot(run_id=best_id, name="acf_pacf")
get_plot(run_id=best_id, name="forecast")
```

### Step D10: Generate Full Run History

```python
all_runs = list_runs(dataset_name="returns", limit=50)
```

This returns all runs ever executed on the "returns" dataset, ordered by creation time. Each entry includes `run_id`, `engine`, `status`, `aic`, `bic`, `duration`, and `created_at`.

---

## Decision Tree: Which Family to Use

```
Is the goal to model volatility/risk?
  Yes --> GARCH
  No  --> Is the goal to detect regime changes?
            Yes --> Markov
            No  --> Is the data a time series with autocorrelation?
                      Yes --> ARIMA
                      No  --> Is the goal to model a relationship between variables?
                                Yes --> OLS
                                No  --> Start with ARIMA as a general-purpose choice
```

## Common Workflow Patterns

### Pattern: Filter by Entity Before Fitting

When your dataset has multiple entities (e.g. multiple stock symbols):

```python
fit_model(
    family="garch",
    target_col="ret",
    query_sql="SELECT date, ret FROM dataset('equities') WHERE symbol = 'AAPL' ORDER BY date",
    params={"p": 1, "q": 1}
)
```

### Pattern: Compute Returns from Prices

```python
fit_model(
    family="garch",
    target_col="log_ret",
    query_sql="""
        SELECT date,
               100 * ln(close / lag(close) OVER (ORDER BY date)) AS log_ret
        FROM dataset('prices')
        WHERE symbol = 'SPY'
        ORDER BY date
    """,
    params={"p": 1, "q": 1}
)
```

### Pattern: Iterative Model Refinement

1. Fit initial model.
2. Check diagnostics -- if Ljung-Box fails, adjust order/params.
3. Re-fit.
4. Compare with previous fit via `compare_models`.
5. Repeat until diagnostics pass and AIC/BIC stop improving.

### Pattern: Post-Hoc Forecast

After fitting without forecasting:

```python
result = fit_model(family="arima", target_col="sales", dataset_name="data", params={"order": [1,1,1]})
# Later, generate a forecast from the same run:
forecast_model(run_id=result["run_id"], steps=12, ci_level=0.95)
```

This only works within the same runtime session because the fitted model object is held in memory.
