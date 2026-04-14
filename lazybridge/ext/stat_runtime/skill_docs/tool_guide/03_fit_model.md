# fit_model Tool Reference

> **Note**: For most analyses, prefer `analyze()` which provides automatic model selection, interpretation, assumptions, and suggested next steps. Use `fit_model` when you need explicit family/parameter control or raw RunRecord output.

## Tool Signature

```python
fit_model(
    family: str,           # Required. "ols", "arima", "garch", or "markov"
    target_col: str,       # Required. Column name for the dependent variable
    dataset_name: str | None = None,   # Registered dataset name
    query_sql: str | None = None,      # SQL query (alternative to dataset_name)
    exog_cols: list[str] | None = None,  # Independent variable columns
    params: dict | None = None,          # Family-specific parameters
    forecast_steps: int | None = None,   # Steps to forecast (None = no forecast)
    time_col: str | None = None,         # Time column for ordering
)
```

## Common Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `family` | `str` | Yes | - | Model family. Must be one of: `"ols"`, `"arima"`, `"garch"`, `"markov"` |
| `target_col` | `str` | Yes | - | The name of the target/dependent variable column in the data |
| `dataset_name` | `str \| None` | No | `None` | Name of a registered dataset. Mutually exclusive with `query_sql`. At least one must be provided. |
| `query_sql` | `str \| None` | No | `None` | SQL query to extract data. Uses `dataset('name')` macro. Mutually exclusive with `dataset_name`. |
| `exog_cols` | `list[str] \| None` | No | `None` | List of exogenous/independent variable column names. Used by OLS and Markov. |
| `params` | `dict \| None` | No | `None` | Family-specific model parameters. See per-family sections below. |
| `forecast_steps` | `int \| None` | No | `None` | Number of out-of-sample forecast steps. `None` or `0` means no forecast. |
| `time_col` | `str \| None` | No | `None` | Time column used for ordering. Not strictly required but recommended for time-series families. |

## Data Sourcing

You must provide exactly one of `dataset_name` or `query_sql`.

### Using dataset_name

Loads the entire registered dataset. The `target_col` and `exog_cols` are extracted as columns.

```python
fit_model(family="ols", target_col="ret", dataset_name="equities", exog_cols=["volume", "spread"])
```

### Using query_sql

Extracts a subset via SQL. The `dataset('name')` macro must be used.

```python
fit_model(
    family="arima",
    target_col="ret",
    query_sql="SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY' ORDER BY date",
    params={"order": [1, 1, 1]}
)
```

The SQL is validated (SELECT only) and executed via DuckDB. NaN values in target and exog columns are automatically removed before fitting.

## Return Value

On success, returns a `RunRecord` dict:

```json
{
  "run_id": "a1b2c3d4e5f6g7h8",
  "dataset_name": "equities",
  "engine": "garch",
  "status": "success",
  "fit_summary": "<model summary text, up to 2000 chars>",
  "params_json": {"mu": 0.05, "omega": 0.01, "alpha[1]": 0.08, "beta[1]": 0.90},
  "metrics_json": {"aic": 1234.56, "bic": 1267.89, "log_likelihood": -612.28},
  "diagnostics_json": [
    {"test_name": "Ljung-Box", "statistic": 8.12, "p_value": 0.62, "passed": true, "interpretation": "..."},
    {"test_name": "Jarque-Bera", "statistic": 3.45, "p_value": 0.18, "passed": true, "interpretation": "..."}
  ],
  "artifact_paths": ["artifacts/.../plots/residuals.png", "artifacts/.../plots/acf_pacf.png", ...],
  "duration_secs": 2.341,
  "created_at": "2025-01-15T10:31:00+00:00",
  "error_message": null
}
```

On failure:

```json
{"error": true, "type": "ValueError", "message": "Target column 'returns' not found. Available: ['date', 'ret', 'volume']"}
```

## Key Metrics in metrics_json

| Metric | Families | Meaning |
|---|---|---|
| `aic` | All | Akaike Information Criterion. Lower = better. Penalizes complexity. |
| `bic` | All | Bayesian Information Criterion. Lower = better. Stronger complexity penalty than AIC. |
| `log_likelihood` | All | Log-likelihood of the fitted model. Higher = better fit. |
| `r_squared` | OLS | Proportion of variance explained. Range 0-1. |
| `adj_r_squared` | OLS | R-squared adjusted for number of predictors. |
| `f_statistic` | OLS | F-test statistic for overall model significance. |
| `f_pvalue` | OLS | P-value for the F-test. < 0.05 means model is significant. |
| `hqic` | ARIMA, Markov | Hannan-Quinn Information Criterion. |

---

## Family: OLS (Ordinary Least Squares)

### When to Use

Linear regression. Use when you need to model a linear relationship between a target variable and one or more explanatory variables.

### Parameters (params dict)

| Key | Type | Default | Description |
|---|---|---|---|
| `add_constant` | `bool` | `true` | Whether to add an intercept term. Set `false` if your exog data already includes a constant. |

### Behavior

- If `exog_cols` is empty (no X provided), OLS fits a trend model: `y ~ constant + time_index`.
- If `exog_cols` is provided and `add_constant` is `true` (default), a constant column is prepended to X.
- Uses `statsmodels.api.OLS`.

### Example: Simple Trend

```python
fit_model(
    family="ols",
    target_col="price",
    dataset_name="stock_prices",
    forecast_steps=30
)
```

### Example: Multiple Regression

```python
fit_model(
    family="ols",
    target_col="ret",
    dataset_name="equities",
    exog_cols=["market_ret", "smb", "hml"],
    params={"add_constant": true}
)
```

### Output params_json

Keys are the regressor names: `"const"`, `"x1"`, `"x2"`, etc. (or original column names if exog provided). Values are coefficient estimates.

### Output extra fields

| Key | Content |
|---|---|
| `p_values` | Dict of regressor name to p-value |
| `std_errors` | Dict of regressor name to standard error |

### Diagnostics Run Automatically

| Test | What It Checks |
|---|---|
| Durbin-Watson | First-order serial correlation in residuals. Ideal near 2.0. |
| Jarque-Bera | Normality of residuals. Null = residuals are normal. |
| Ljung-Box | Serial correlation at multiple lags. Null = no autocorrelation. |

### Plots Generated

- `residuals` -- scatter plot + histogram of residuals
- `acf_pacf` -- ACF and PACF of residuals
- `forecast` -- (only if `forecast_steps` is set) trend forecast with confidence bands

---

## Family: ARIMA (AutoRegressive Integrated Moving Average)

### When to Use

Time-series modeling with autoregressive, differencing, and moving average components. Use for stationary or near-stationary univariate series. For seasonal data, use the seasonal_order parameter (SARIMAX).

### Parameters (params dict)

| Key | Type | Default | Description |
|---|---|---|---|
| `order` | `list[int]` (length 3) | `[1, 0, 0]` | ARIMA order as `[p, d, q]`. `p` = AR lags, `d` = differencing order, `q` = MA lags. |
| `seasonal_order` | `list[int]` (length 4) | `[0, 0, 0, 0]` | Seasonal ARIMA order as `[P, D, Q, s]`. `s` = seasonal period (e.g. 12 for monthly). |
| `trend` | `str` | `"c"` | Trend component: `"n"` (none), `"c"` (constant), `"t"` (linear trend), `"ct"` (constant + trend). |
| `enforce_stationarity` | `bool` | `false` | Whether to enforce stationarity on AR parameters. |
| `enforce_invertibility` | `bool` | `false` | Whether to enforce invertibility on MA parameters. |

### Behavior

- Uses `statsmodels.tsa.statespace.sarimax.SARIMAX`.
- Exogenous variables (`exog_cols`) are supported (ARIMAX/SARIMAX).
- Fitted with `disp=False` (no console output).

### Example: ARIMA(1,1,1)

```python
fit_model(
    family="arima",
    target_col="price",
    query_sql="SELECT date, price FROM dataset('stocks') WHERE symbol = 'AAPL' ORDER BY date",
    params={"order": [1, 1, 1]},
    forecast_steps=20
)
```

### Example: Seasonal ARIMA

```python
fit_model(
    family="arima",
    target_col="sales",
    dataset_name="monthly_sales",
    params={
        "order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 12],
        "trend": "c"
    },
    forecast_steps=12
)
```

### Example: ARIMAX with Exogenous Variables

```python
fit_model(
    family="arima",
    target_col="gdp_growth",
    dataset_name="macro",
    exog_cols=["interest_rate", "inflation"],
    params={"order": [2, 0, 1]}
)
```

### Output params_json

Keys follow statsmodels naming: `"ar.L1"`, `"ar.L2"`, `"ma.L1"`, `"sigma2"`, `"intercept"`, etc.

### Output extra fields

| Key | Content |
|---|---|
| `order` | The ARIMA order used, as a list `[p, d, q]` |
| `seasonal_order` | The seasonal order used, as a list `[P, D, Q, s]` |

### Diagnostics Run Automatically

| Test | What It Checks |
|---|---|
| Ljung-Box | Serial correlation in residuals. Lags = min(10, n/5). |
| Jarque-Bera | Normality of residuals. |

### Plots Generated

- `residuals` -- scatter plot + histogram of residuals
- `acf_pacf` -- ACF and PACF of residuals (useful for checking if order is adequate)
- `forecast` -- (only if `forecast_steps` is set) point forecast with confidence bands

---

## Family: GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

### When to Use

Modeling time-varying volatility. Use for financial return series that exhibit volatility clustering. The target column should be returns (not prices).

### Parameters (params dict)

| Key | Type | Default | Description |
|---|---|---|---|
| `p` | `int` | `1` | GARCH lag order (number of lagged conditional variance terms). |
| `q` | `int` | `1` | ARCH lag order (number of lagged squared residual terms). |
| `vol` | `str` | `"GARCH"` | Volatility model: `"GARCH"`, `"EGARCH"`, `"TARCH"`, `"FIGARCH"`, `"HARCH"`, `"APARCH"`. |
| `dist` | `str` | `"normal"` | Error distribution: `"normal"`, `"t"`, `"skewt"`, `"ged"`. |
| `mean` | `str` | `"Constant"` | Mean model: `"Constant"`, `"Zero"`, `"AR"`, `"ARX"`, `"HAR"`, `"LS"`. |
| `rescale` | `bool` | `false` | Whether to rescale data (multiply by 100). Set `true` if returns are in decimal form (e.g. 0.01 for 1%). |

### Behavior

- Uses the `arch` library's `arch_model` function.
- Fitted with `disp="off"` (no console output).
- `exog_cols` is NOT used by GARCH. The target column should contain the return series.
- The `fitted_values_json` field contains the conditional volatility series (not typical fitted values).

### Example: Standard GARCH(1,1)

```python
fit_model(
    family="garch",
    target_col="ret",
    dataset_name="equities",
    params={"p": 1, "q": 1},
    forecast_steps=20
)
```

### Example: EGARCH with Student-t Errors

```python
fit_model(
    family="garch",
    target_col="ret",
    query_sql="SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY' ORDER BY date",
    params={"p": 1, "q": 1, "vol": "EGARCH", "dist": "t"}
)
```

### Example: GJR-GARCH (TARCH) for Asymmetric Volatility

```python
fit_model(
    family="garch",
    target_col="ret",
    dataset_name="equities",
    params={"p": 1, "q": 1, "vol": "TARCH", "dist": "skewt"}
)
```

### Output params_json

Keys follow `arch` library naming:

| Key | Meaning |
|---|---|
| `mu` | Mean model constant |
| `omega` | Variance intercept |
| `alpha[1]` | ARCH coefficient (impact of past shocks) |
| `beta[1]` | GARCH coefficient (persistence of volatility) |
| `nu` | Degrees of freedom (if `dist="t"` or `"skewt"`) |
| `lambda` | Skewness parameter (if `dist="skewt"`) |
| `gamma[1]` | Asymmetry parameter (if `vol="EGARCH"` or `"TARCH"`) |

### Output extra fields

| Key | Content |
|---|---|
| `p` | GARCH lag order used |
| `q` | ARCH lag order used |
| `vol_model` | Volatility model name |
| `distribution` | Error distribution name |
| `conditional_volatility` | List of floats: time-varying volatility estimate for each observation |
| `std_residuals` | Standardized residuals (residuals / conditional_volatility) |
| `p_values` | Dict of parameter name to p-value |

### Interpreting GARCH Output

- `alpha[1] + beta[1]` close to 1.0 means high volatility persistence.
- `alpha[1]` captures the reaction to market shocks.
- `beta[1]` captures the persistence of past volatility.
- If `alpha[1] + beta[1] >= 1.0`, the process is IGARCH (integrated GARCH) -- volatility shocks are permanent.

### Forecast Output (extra fields)

When `forecast_steps` is set, the `ForecastResult` includes:

| Key | Content |
|---|---|
| `point_forecast` | Forecasted mean returns |
| `lower_ci` / `upper_ci` | Confidence bounds using forecasted volatility |
| `variance_forecast` | Forecasted conditional variance per step |
| `volatility_forecast` | Forecasted conditional volatility (sqrt of variance) per step |

### Diagnostics Run Automatically

| Test | What It Checks |
|---|---|
| Ljung-Box (standardized residuals) | Serial correlation in standardized residuals. |
| Ljung-Box (squared residuals) | Remaining ARCH effects in squared standardized residuals. Should pass if GARCH is adequate. |
| Jarque-Bera | Normality of standardized residuals. |

### Plots Generated

- `residuals` -- scatter + histogram of raw residuals
- `acf_pacf` -- ACF/PACF of residuals
- `volatility` -- conditional volatility with returns overlay (dual-axis plot)
- `forecast` -- (only if `forecast_steps` is set) forecast with CI bands

---

## Family: Markov (Markov Switching Regression)

### When to Use

Regime-switching models. Use when the data-generating process is believed to switch between distinct states (e.g. bull/bear markets, expansion/recession, high/low volatility).

### Parameters (params dict)

| Key | Type | Default | Description |
|---|---|---|---|
| `k_regimes` | `int` | `2` | Number of distinct regimes. Typically 2 or 3. |
| `order` | `int` | `0` | Autoregressive order within each regime. `0` = no AR terms. |
| `trend` | `str` | `"c"` | Trend component: `"n"` (none), `"c"` (constant), `"t"` (trend), `"ct"` (both). |
| `switching_variance` | `bool` | `true` | Whether variance differs across regimes. Set `true` for volatility regime detection. |

### Behavior

- Uses `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`.
- Exogenous variables (`exog_cols`) ARE supported.
- Fitted with `disp=False`.
- The model estimates regime-specific intercepts and (optionally) variances, plus a transition probability matrix.

### Example: 2-Regime Switching Mean

```python
fit_model(
    family="markov",
    target_col="ret",
    dataset_name="equities",
    params={"k_regimes": 2, "switching_variance": true}
)
```

### Example: 3-Regime with AR(1)

```python
fit_model(
    family="markov",
    target_col="gdp_growth",
    dataset_name="macro",
    params={"k_regimes": 3, "order": 1, "switching_variance": true},
    forecast_steps=8
)
```

### Example: With Exogenous Variables

```python
fit_model(
    family="markov",
    target_col="ret",
    dataset_name="equities",
    exog_cols=["vix"],
    params={"k_regimes": 2}
)
```

### Output params_json

Keys follow statsmodels naming for Markov models:

| Key Pattern | Meaning |
|---|---|
| `const[0]`, `const[1]` | Regime-specific intercepts |
| `sigma2[0]`, `sigma2[1]` | Regime-specific variances (if `switching_variance=true`) |
| `p[0->0]`, `p[1->0]` | Transition probabilities |
| `ar.L1[0]`, `ar.L1[1]` | Regime-specific AR coefficients (if `order > 0`) |

### Output extra fields

| Key | Content |
|---|---|
| `k_regimes` | Number of regimes |
| `transition_matrix` | List of lists: `transition_matrix[i][j]` = P(regime j at t+1 \| regime i at t) |
| `smoothed_probabilities` | Dict `{"regime_0": [...], "regime_1": [...]}`. Each value is a list of floats (probability at each time step). |
| `regime_info` | Dict with `regime_0_duration`, `regime_1_duration` etc. Expected duration in each regime (in periods). |

### Interpreting Markov Output

- **Transition matrix**: `transition_matrix[i][j]` is the probability of moving from regime i to regime j. Diagonal values close to 1.0 mean regimes are "sticky" (long-lasting).
- **Smoothed probabilities**: Time-varying probability of being in each regime. Use `get_plot(run_id, "regimes")` to visualize.
- **Expected duration**: `1 / (1 - p[i->i])`. A regime with `p[0->0] = 0.95` has an expected duration of 20 periods.

### Forecast Behavior

Markov forecasts use regime-weighted prediction:
1. Start from the last smoothed regime probabilities.
2. At each step, multiply by the transition matrix to get the next regime probabilities.
3. The point forecast is the weighted average of regime-specific means.
4. Confidence intervals use the overall residual standard deviation.

### Diagnostics Run Automatically

| Test | What It Checks |
|---|---|
| Ljung-Box | Serial correlation in residuals. |
| Jarque-Bera | Normality of residuals. |
| Regime Classification Certainty | Average max probability across all time steps. > 0.70 = good separation. |

### Plots Generated

- `residuals` -- scatter + histogram
- `acf_pacf` -- ACF/PACF of residuals
- `regimes` -- multi-panel plot: observed series (top) + smoothed probability for each regime (below)
- `forecast` -- (only if `forecast_steps` is set) regime-weighted forecast with CI bands

---

## Choosing Between dataset_name and query_sql

| Use Case | Approach |
|---|---|
| Fit on the entire dataset | `dataset_name="my_data"` |
| Filter to a specific symbol or date range | `query_sql="SELECT ... FROM dataset('my_data') WHERE ..."` |
| Join multiple datasets | `query_sql="SELECT a.ret, b.vix FROM dataset('returns') a JOIN dataset('vix') b ON a.date = b.date"` |
| Compute derived columns | `query_sql="SELECT date, ln(close/lag(close) OVER (ORDER BY date)) AS log_ret FROM dataset('prices')"` |

Always add `ORDER BY <time_col>` when using `query_sql` with time-series models to ensure correct temporal ordering.

## Automatic Behavior After Fitting

Every successful `fit_model` call automatically:

1. **Runs diagnostics** -- family-specific tests stored in `diagnostics_json`.
2. **Generates plots** -- residuals and ACF/PACF for all families, plus volatility (GARCH), regimes (Markov), forecast (if requested).
3. **Saves artifacts** -- spec JSON, fit summary text, diagnostics JSON, residual data, forecast data.
4. **Persists the run record** -- retrievable via `get_run(run_id)` and `list_runs()`.

## Error Handling

The tool never raises exceptions. All errors are returned as:

```json
{"error": true, "type": "ExceptionType", "message": "Detailed error message"}
```

Common errors:
- `"ModelSpec must have either dataset_name or query_sql"` -- neither data source provided.
- `"Target column 'X' not found. Available: [...]"` -- wrong column name.
- `"Dataset 'X' is not registered"` -- dataset name not found in catalog.
- `"Query returned no data"` -- SQL returned 0 rows.
- Convergence errors from statsmodels/arch -- check data quality, try different params.
