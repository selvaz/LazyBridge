# Diagnostics Tool Reference

## Available Diagnostic Tests

### Stationarity Tests (via run_diagnostics tool)

#### Augmented Dickey-Fuller (ADF)
- Null hypothesis: series has a unit root (non-stationary)
- p-value < 0.05 → reject null → series IS stationary
- Use before ARIMA/GARCH fitting

#### KPSS
- Null hypothesis: series IS stationary
- p-value < 0.05 → reject null → series is NOT stationary
- Use alongside ADF for confirmation

### Residual Diagnostics (auto-run after fit)

#### Ljung-Box
- Tests for serial correlation in residuals
- p-value > 0.05 → no significant autocorrelation (good)
- Applied to both raw and squared residuals for GARCH models

#### Jarque-Bera
- Tests for normality of residuals
- p-value > 0.05 → residuals approximately normal
- Reports skewness and kurtosis

#### Durbin-Watson
- Tests for first-order autocorrelation
- Value near 2 = no autocorrelation
- Below 1.5 = positive autocorrelation (bad)
- Above 2.5 = negative autocorrelation

### Model Comparison (via compare_models tool)
- Compares AIC and BIC across multiple runs
- Lower AIC/BIC = better model
- Identifies the best model by each criterion

## Tool Usage

### Pre-fit stationarity check
```
run_diagnostics(series_name="equities", column="ret")
```
Returns list of ADF and KPSS results with interpretations.

### Post-fit diagnostics
Diagnostics run automatically when you call fit_model. Check:
```
result = get_run(run_id)
result["diagnostics_json"]  # list of diagnostic results
```

### Model comparison
```
compare_models(run_ids=["run1_id", "run2_id", "run3_id"])
```

## Reading Results

Each diagnostic result contains:
- `test_name`: Name of the test
- `statistic`: Test statistic value
- `p_value`: P-value (if applicable)
- `passed`: Whether the test passed at 5% level
- `interpretation`: Human-readable explanation

## Decision Framework
1. Run stationarity tests FIRST
2. If non-stationary: difference the series or use ARIMA with d>0
3. After fitting: check that residual diagnostics pass
4. If Ljung-Box fails: model may need more AR/MA terms
5. If Jarque-Bera fails: consider non-normal distribution (t-dist for GARCH)
6. Compare multiple specifications using AIC/BIC
