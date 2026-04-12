# Errors, Diagnostics, and Recovery

## Error Response Format

All stat_runtime tools catch exceptions internally and return error dicts instead of raising. Every error response has this shape:

```json
{
  "error": true,
  "type": "ExceptionType",
  "message": "Detailed error message"
}
```

For tools that return lists (e.g. `list_datasets`, `run_diagnostics`, `list_runs`), errors are wrapped in a single-element list:

```json
[{"error": true, "type": "ExceptionType", "message": "..."}]
```

For `fit_model`, a failed run still returns a full `RunRecord` dict with `status="failed"` and `error_message` set, rather than the error dict. This preserves the run ID for debugging.

## Detection Pattern

Always check for errors after each tool call:

```python
result = fit_model(...)

# Pattern 1: fit_model returns RunRecord even on failure
if result.get("status") == "failed":
    error_msg = result["error_message"]
    # Handle the error

# Pattern 2: Other tools return error dicts
result = register_dataset(...)
if result.get("error"):
    error_type = result["type"]
    error_msg = result["message"]
    # Handle the error
```

---

## Missing Dependency Errors

### ImportError: duckdb/polars/statsmodels/arch/matplotlib

**Error message**: `"duckdb is required for this feature. Run: pip install lazybridge[stats]"`

**Cause**: The `lazybridge[stats]` extras are not installed. The stat_runtime uses lazy imports -- the missing package is only detected when the specific feature is called.

**Which tools trigger which dependency**:

| Dependency | Required By |
|---|---|
| `polars` | `register_dataset`, `profile_dataset`, `fit_model` (when using `dataset_name`) |
| `duckdb` | `query_data`, `fit_model` (when using `query_sql`) |
| `sqlglot` | `query_data` (SQL validation) |
| `statsmodels` | `fit_model` (OLS, ARIMA, Markov), `run_diagnostics`, all ACF/PACF plots |
| `arch` | `fit_model` (GARCH only) |
| `matplotlib` | All plot generation (non-fatal -- fit succeeds but plots are skipped) |
| `scipy` | `forecast_model` (GARCH and Markov CI computation) |

**Recovery**: Run `pip install lazybridge[stats]` to install all dependencies at once. Or install individual packages: `pip install duckdb polars sqlglot statsmodels arch matplotlib scipy`.

**Check availability programmatically**:

```python
from lazybridge.stat_runtime._deps import is_available, STATS_AVAILABLE

# Check all stats deps at once
if not STATS_AVAILABLE:
    print("Missing stats dependencies")

# Check individual packages
is_available("arch")       # True/False
is_available("statsmodels") # True/False
```

---

## Dataset Registration Errors

### FileNotFoundError: Parquet/CSV path does not exist

**Error message**: `"Parquet path does not exist: /data/missing_file.parquet"`

**Cause**: The `uri` parameter points to a file that does not exist on disk.

**Recovery**: Verify the file path. Use an absolute path. Check permissions.

### ValueError: time_column not found in schema

**Error message**: `"time_column 'timestamp' not found in schema. Available columns: ['date', 'symbol', 'ret', 'volume']"`

**Cause**: The `time_column` parameter does not match any column name in the file.

**Recovery**: Check the exact column name. Column names are case-sensitive. Use `discover_data()` or check the `columns_schema` in the registration response to see exact column names.

### Dataset name collision

Registering a dataset with the same name as an existing one will overwrite the previous registration. This is by design -- no error is raised. The old metadata is replaced.

---

## Unregistered Dataset Errors

### ValueError: Dataset 'X' is not registered

**Error message**: `"Dataset 'equities' is not registered. Available: ['stocks', 'returns']"`

**Cause**: The dataset name used in `fit_model(dataset_name=...)`, `query_data(sql="...dataset('name')...")`, or `profile_dataset(name=...)` does not match any registered dataset.

**Recovery**:
1. Call `list_datasets()` to see all registered dataset names.
2. Check for typos in the dataset name. Names are exact-match, case-sensitive.
3. If no datasets are registered, call `register_dataset` first.

### ValueError: Dataset not registered (in query_sql)

**Error message**: `"Dataset 'prices' is not registered. Available: ['equities']"`

**Cause**: The SQL in `query_sql` uses `dataset('prices')` but no dataset named `"prices"` exists.

**Recovery**: Register the dataset first, or fix the name in the SQL macro.

---

## SQL Query Errors

### ValueError: Only SELECT statements allowed

**Error message**: `"Only SELECT statements are allowed. Got: INSERT... INSERT, UPDATE, DELETE, DROP, CREATE, and other mutations are blocked."`

**Cause**: The SQL passed to `query_data` or used in `fit_model(query_sql=...)` is not a SELECT (or WITH) statement.

**Recovery**: Rewrite the query as a SELECT statement. The query engine only supports read operations.

### ValueError: Forbidden SQL keyword

**Error message**: `"Forbidden SQL keyword detected: DROP. Only SELECT queries are allowed."`

**Cause**: The SQL contains a blocked keyword (INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, COPY, EXPORT, IMPORT, LOAD, read_csv_auto, read_json_auto).

**Recovery**: Remove the forbidden keyword. Use only SELECT/WITH queries. For reading data, use the `dataset('name')` macro instead of `read_csv_auto()` or `read_json_auto()`.

### DuckDB execution errors

**Error message**: varies, often `"Catalog Error: Table ... does not exist"` or `"Binder Error: column ... not found"`

**Cause**: The SQL has a syntax error, references non-existent columns, or has type mismatches.

**Recovery**:
1. Verify column names by calling `list_datasets()` or `profile_dataset(name=...)`.
2. Use the `dataset('name')` macro -- raw table names will not work.
3. Check SQL syntax. DuckDB uses standard SQL.
4. Test with a simple query first: `SELECT * FROM dataset('name') LIMIT 5`.

### Query returns no data

**Error message**: `"Query returned no data"`

**Cause**: The WHERE clause filtered out all rows, or the dataset is empty.

**Recovery**: Remove or relax the WHERE clause. Check row counts with `list_datasets()`.

---

## Data Quality Errors

### ValueError: Target column not found

**Error message**: `"Target column 'returns' not found. Available: ['date', 'ret', 'volume']"`

**Cause**: The `target_col` parameter does not match any column in the data.

**Recovery**: Check the exact column name. If using `dataset_name`, the columns come from the registered schema. If using `query_sql`, the columns come from the query's SELECT clause.

### NaN handling

NaN values in the target and exogenous columns are automatically stripped before fitting. Rows where `target_col` is NaN or any `exog_cols` value is NaN are removed. This is silent -- no error is raised.

If ALL values are NaN, the resulting array is empty and you will get a convergence or dimension error from the underlying engine.

### Empty data after NaN removal

**Error message**: varies -- typically a numpy or statsmodels error about array dimensions.

**Recovery**: Profile the data first to check `null_pct`. If the target column is mostly nulls, fix the data or use a different column.

---

## Non-Stationary Data Errors

### Symptom: ARIMA convergence failure on non-stationary data

**Error message**: varies -- may be `"MLE optimization failed to converge"` or `"LinAlgError: Singular matrix"`.

**Cause**: The data has a unit root (trend or random walk) and the ARIMA model specification does not include differencing (`d=0`).

**Recovery**:
1. Run `run_diagnostics(series_name="...", column="...")` to check stationarity.
2. If ADF test fails (non-stationary), increase the differencing order: `"order": [p, 1, q]`.
3. If the series has a strong trend, consider using `"trend": "ct"` or differencing.

### Symptom: GARCH fit on price levels

**Error message**: Often no explicit error, but the model output will show `alpha[1] + beta[1] >= 1.0` (IGARCH) and diagnostics will fail.

**Cause**: GARCH should be fitted on returns, not price levels. Price levels are non-stationary.

**Recovery**: Compute returns before fitting. Use a SQL query:

```python
fit_model(
    family="garch",
    target_col="ret",
    query_sql="SELECT date, 100 * (close - lag(close) OVER (ORDER BY date)) / lag(close) OVER (ORDER BY date) AS ret FROM dataset('prices') ORDER BY date",
    params={"p": 1, "q": 1}
)
```

### Interpreting ADF and KPSS Together

| ADF Result | KPSS Result | Interpretation | Action |
|---|---|---|---|
| Stationary (passed) | Stationary (passed) | Series is stationary | Proceed with `d=0` |
| Non-stationary (failed) | Stationary (passed) | Conflicting -- borderline | Try `d=0` first, then `d=1` |
| Stationary (passed) | Non-stationary (failed) | Conflicting -- trend-stationary | Use `trend="ct"` in model |
| Non-stationary (failed) | Non-stationary (failed) | Series is non-stationary | Use `d=1` (or `d=2` if `d=1` is insufficient) |

---

## Convergence Failures

### ConvergenceWarning / MLE Optimization Failed

**Error message**: `"ConvergenceWarning: Maximum Likelihood optimization failed to converge"` or similar.

**Cause**: The optimizer could not find a good fit. Common reasons:
- Data is too short (< 100 observations for GARCH, < 50 for ARIMA).
- Model is over-specified for the data (too many parameters relative to data).
- Data has extreme outliers that distort the likelihood surface.
- Starting values are poor for the optimization.

**Recovery strategies**:

1. **Simplify the model**: Reduce `p`, `q`, or `k_regimes`. Use fewer exogenous variables.
2. **Check data length**: GARCH typically needs 250+ daily observations. ARIMA needs 50+. Markov needs 200+.
3. **Check for outliers**: Profile the data and look for extreme `min_val` or `max_val`.
4. **Try different specifications**:
   - For GARCH: change `mean` from `"Constant"` to `"Zero"`, or change `dist` to `"t"`.
   - For ARIMA: try `enforce_stationarity=false` and `enforce_invertibility=false` (the defaults).
   - For Markov: reduce `k_regimes` or set `switching_variance=false`.
5. **Rescale data**: For GARCH, if returns are in decimal form (0.001 instead of 0.1%), set `rescale=true` or multiply by 100 in the SQL query.

### LinAlgError: Singular matrix

**Cause**: The data matrix is singular (perfectly collinear columns, or near-zero variance).

**Recovery**:
1. Check for constant columns in the data.
2. For OLS: check that exogenous variables are not perfectly correlated. Remove one of any pair of highly correlated variables.
3. For ARIMA/Markov: check that the data has variance (not a constant series).

### arch: Optimization result indicates convergence issues

**Cause**: The `arch` library's optimizer did not converge cleanly. The model may still produce results, but they should be treated with caution.

**Recovery**: Try different starting parameters, a different volatility model (`vol`), or a different distribution (`dist`).

---

## Forecast Errors

### ValueError: forecast requires a fitted model object (re-fit first)

**Error message**: `"GARCH forecast requires a fitted model object (re-fit first)"`

**Cause**: `forecast_model` was called with a `run_id` from a previous session. The fitted model object (held in memory) is no longer available.

**Recovery**: Re-fit the model using `fit_model` with the same parameters, then use the new `run_id` for forecasting. Alternatively, include `forecast_steps` in the original `fit_model` call.

### ValueError: Run 'X' not found

**Error message**: `"Run 'abc123' not found"`

**Cause**: The `run_id` does not exist in the MetaStore.

**Recovery**: Call `list_runs()` to find valid run IDs. Check for typos.

### ValueError: Run did not succeed

**Error message**: `"Run 'abc123' did not succeed (status=failed)"`

**Cause**: Attempting to forecast from a failed run.

**Recovery**: Check the run's error message with `get_run(run_id=...)`. Fix the underlying issue and re-fit.

---

## Runtime Initialization Errors

### RuntimeError: stat_runtime tools not initialized

**Error message**: `"stat_runtime tools not initialized. Use stat_tools(runtime) first."`

**Cause**: Tool functions were called before `stat_tools(runtime)` was invoked. The module-level `_runtime` reference is `None`.

**Recovery**: Create a `StatRuntime` instance and call `stat_tools(rt)` before using any tools.

```python
from lazybridge.stat_runtime.runner import StatRuntime
from lazybridge.stat_runtime.tools import stat_tools

rt = StatRuntime()
tools = stat_tools(rt)
# Now tools are ready
```

---

## Diagnostic Test Interpretation

When a diagnostic test has `"passed": false`, it does not necessarily mean the model is invalid. Here is how to interpret each test failure:

### Ljung-Box Failed (Serial Correlation Detected)

**Meaning**: Residuals have statistically significant autocorrelation. The model has not captured all temporal structure.

**Severity**: High for ARIMA (the model's core purpose is temporal modeling). Moderate for GARCH (check squared residuals test instead). Low for OLS (may not be a time-series model).

**Fix**:
- For ARIMA: increase `p` or `q` in the order.
- For GARCH: this tests raw residuals. Check the squared residuals Ljung-Box separately -- if that passes, the GARCH model is adequate for volatility even if the mean model is misspecified.
- For Markov: increase `order` (AR terms within regimes).

### Ljung-Box on Squared Residuals Failed (GARCH only)

**Meaning**: There are remaining ARCH effects that the GARCH model did not capture.

**Severity**: High -- the volatility model is inadequate.

**Fix**: Increase `p` or `q`, try an asymmetric model (`vol="EGARCH"` or `vol="TARCH"`), or try a different distribution (`dist="t"`).

### Jarque-Bera Failed (Non-Normal Residuals)

**Meaning**: Residuals are not normally distributed (heavy tails, skewness, or both).

**Severity**: Low for point forecasting. Moderate for confidence intervals (CIs may be too narrow/wide).

**Fix**: For GARCH, use `dist="t"` or `dist="skewt"` to accommodate fat tails. For ARIMA/OLS, this is informational -- point forecasts are still valid, but CIs based on normality assumptions may be unreliable.

### Durbin-Watson Failed (OLS only)

**Meaning**: First-order autocorrelation in OLS residuals. The errors are not independent.

**Severity**: High -- OLS standard errors and p-values are unreliable.

**Fix**: Consider using ARIMA instead of OLS. Or add lagged dependent variables to the OLS model. Or use Newey-West (HAC) standard errors (not directly available in the tool interface -- this requires programmatic access).

### Regime Classification Certainty Failed (Markov only)

**Meaning**: The model cannot clearly distinguish between regimes. Average max regime probability across all time steps is below 0.70.

**Severity**: Moderate -- the regimes may not be economically meaningful.

**Fix**: Reduce `k_regimes`. Try `switching_variance=true` if not already set. Check if the data actually exhibits regime-switching behavior.

---

## Common Mistakes and Corrections

| Mistake | Symptom | Correction |
|---|---|---|
| Using price levels for GARCH | IGARCH (alpha+beta >= 1), poor diagnostics | Use returns instead of prices |
| Forgetting to register dataset | `"Dataset 'X' is not registered"` | Call `register_dataset` first |
| Wrong column name | `"Target column 'X' not found"` | Check `list_datasets()` for exact column names |
| ARIMA on non-stationary data | Convergence failure or unreliable forecast | Increase `d` in the order, or difference the data |
| Too many parameters for data size | Convergence failure | Reduce model complexity (lower p, q, k_regimes) |
| Forecasting from a dead session | `"requires a fitted model object"` | Re-fit the model or use `forecast_steps` during fit |
| Missing ORDER BY in time-series SQL | Data fed to model in arbitrary order | Add `ORDER BY date` to SQL |
| Using `dataset_name` AND `query_sql` | Only one is used (dataset_name takes priority in data loading, but behavior may be confusing) | Use exactly one. Prefer `query_sql` for filtered data. |
| Fitting GARCH on returns in decimal form (0.001) | Very small parameter estimates, possible numerical issues | Set `rescale=true` or multiply returns by 100 in SQL |

---

## Debugging Checklist

When a `fit_model` call returns `status="failed"`:

1. Read `error_message` in the returned dict.
2. Check if it is a dependency error (ImportError) -- install packages.
3. Check if it is a data error (column not found, no data) -- verify dataset and column names.
4. Check if it is a convergence error -- simplify the model or fix data issues.
5. Profile the data: `profile_dataset(name=...)` to check nulls, ranges, row counts.
6. Run stationarity tests: `run_diagnostics(series_name=..., column=...)`.
7. Try a simpler model first (e.g. ARIMA(1,0,0) or GARCH(1,1) with defaults).
8. Check data length -- ensure enough observations for the model complexity.
