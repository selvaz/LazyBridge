# Forecast Evaluation

## In-Sample vs Out-of-Sample

### The Fundamental Rule
In-sample fit metrics (R-squared, log-likelihood, AIC) measure how well the model explains data it was trained on. Out-of-sample metrics measure how well it predicts unseen data. For financial applications, **out-of-sample performance is what matters**.

### Why In-Sample Misleads
- A model with enough parameters can memorize noise. In-sample R-squared always increases with more parameters.
- If you tested 20 specifications and picked the best in-sample, you are implicitly overfitting.
- Financial data has regime shifts. A model that fits 2005-2015 perfectly may fail in 2016-2020.

## Out-of-Sample Testing for Time Series

### Expanding Window (preferred for most applications)
1. Fit the model on data from time 1 to time T.
2. Forecast time T+1.
3. Expand: fit on time 1 to T+1.
4. Forecast T+2.
5. Continue until end of sample.

### Rolling Window (preferred when structural change is suspected)
1. Fit on a fixed window of W observations.
2. Forecast the next observation.
3. Roll: drop the oldest, add the newest.
4. Continue until end of sample.

### Design Choices
- Reserve at least 20-30% of data for out-of-sample evaluation.
- Daily data, 2500 obs (10 years): first 1500 training, last 1000 evaluation.
- Monthly data, 360 obs (30 years): first 240 training, last 120 evaluation.
- Rolling window size: 5-10 years for daily, 15-20 years for monthly.

## Forecast Accuracy Metrics

### Point Forecast Metrics

| Metric | Formula | Use Case |
|---|---|---|
| MAE | mean(\|e_t\|) | Robust to outliers, easy to interpret |
| RMSE | sqrt(mean(e_t²)) | Penalizes large errors more heavily |
| MAPE | mean(\|e_t / y_t\|) × 100 | Scale-free, but undefined when y_t = 0 |
| Directional accuracy | mean(sign(fc_t) == sign(y_t)) | For trading signal evaluation |
| Out-of-sample R² | 1 - sum(e_t²) / sum((y_t - ȳ)²) | Improvement over historical mean |

### What to Expect

**Return forecasting**: Out-of-sample R² of 0.5%-3.0% is considered economically meaningful (Campbell & Thompson, 2008). Do not expect high R² for return prediction. A daily return forecasting R² of 1% can generate substantial trading profits after accounting for transaction costs.

**Volatility forecasting**: R² of 20%-50% against realized volatility is typical for good GARCH models. GARCH consistently outperforms historical volatility for 1-5 day horizons.

**Level forecasting** (GDP, prices): RMSE relative to a random walk benchmark matters more than absolute R². Beating the random walk is the baseline test.

## Comparing Two Forecasts

### Diebold-Mariano Test
Tests whether two forecast methods have equal predictive accuracy.

- Null hypothesis: both methods are equally accurate.
- If p < 0.05: the forecasts are significantly different in accuracy.
- Uses the loss differential series: d_t = L(e_{1,t}) - L(e_{2,t}), where L is a loss function (usually squared error or absolute error).
- Accounts for serial correlation in the loss differentials.

### When Diebold-Mariano is Not Applicable
- Nested models (e.g., GARCH(1,1) vs GARCH(2,1)): use the Clark-West test instead.
- Very short evaluation periods (< 50 observations): power is low.
- Non-stationary loss differentials: results unreliable.

## Forecast Combination

### Simple Average Often Wins
Combining forecasts from multiple models frequently outperforms any individual model. The simplest approach — equal-weight average — is surprisingly hard to beat.

**Why it works**: different models capture different aspects of the data. Their errors are partially uncorrelated, so averaging reduces overall forecast variance.

### Practical Rules
- Combine 2-4 models maximum. More than 4 rarely improves.
- Equal weights are a strong baseline. Estimated optimal weights tend to overfit.
- Trim extreme forecasts before averaging if models occasionally diverge wildly.

## Forecast Horizon Effects

### Short Horizon (1-5 steps)
- Model-specific dynamics dominate.
- GARCH volatility forecasts are most accurate here.
- ARIMA mean forecasts degrade quickly.

### Medium Horizon (5-20 steps)
- Forecast uncertainty grows substantially.
- Confidence intervals widen (roughly proportional to sqrt(horizon)).
- Model differences diminish as all forecasts revert toward unconditional means.

### Long Horizon (>20 steps)
- Most model-based forecasts converge to the unconditional mean/variance.
- Use unconditional estimates or structural models instead.
- GARCH volatility forecasts beyond 20 days are essentially the unconditional volatility.

## Common Mistakes

1. **Evaluating forecasts in-sample**: Always use out-of-sample evaluation. In-sample accuracy is not predictive accuracy.
2. **Choosing models by in-sample R²**: Use information criteria (AIC/BIC) or out-of-sample metrics.
3. **Not accounting for look-ahead bias**: In expanding/rolling window evaluation, make sure you only use data available at each forecast origin.
4. **Comparing forecasts on different samples**: All forecasts being compared must cover the exact same out-of-sample period.
5. **Ignoring forecast combination**: Testing 10 models and picking the best is inferior to combining the top 3.
6. **Expecting high R² for returns**: Return predictability is inherently low. R² of 1-2% is good, not bad.
7. **Using MAPE when values cross zero**: MAPE is undefined or misleading for series that cross zero (like returns). Use MAE or RMSE instead.
