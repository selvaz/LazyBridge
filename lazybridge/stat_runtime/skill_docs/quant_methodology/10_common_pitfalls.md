# Common Statistical Pitfalls

## 1. Spurious Regression
**What**: Two non-stationary series appear correlated by chance.
**Example**: GDP and number of movies produced both trend upward — OLS will show a "significant" relationship that doesn't exist.
**Fix**: Test for stationarity (ADF, KPSS). Use differenced data or cointegration tests. Check if R² is higher than Durbin-Watson statistic — a classic signal.

## 2. Data Snooping / P-Hacking
**What**: Running many specifications until one produces significant results.
**Example**: Testing 100 trading strategies, finding 5 that "work" at p<0.05 — expected by chance alone.
**Fix**: Pre-specify hypotheses. Use out-of-sample validation. Apply Bonferroni correction for multiple comparisons. Report ALL specifications tested.

## 3. Survivorship Bias
**What**: Analyzing only entities that survived to the present.
**Example**: Studying stock returns using only currently listed companies — excludes delisted/bankrupt firms, biasing returns upward.
**Fix**: Use point-in-time data. Include delisted securities. Be explicit about the sample construction.

## 4. Look-Ahead Bias
**What**: Using information that wasn't available at the time of the decision.
**Example**: Using revised GDP data to backtest a strategy when only preliminary estimates were available in real time.
**Fix**: Use only point-in-time data. Ensure all features in the model were available at each historical point.

## 5. Non-Stationarity
**What**: Statistical properties (mean, variance) change over time.
**Example**: Running OLS on price levels instead of returns. t-statistics are unreliable, R² is inflated.
**Fix**: Test with ADF + KPSS. Difference the data. Use ARIMA with d>0. For volatility, use GARCH on returns.

## 6. Multicollinearity
**What**: Independent variables are highly correlated with each other.
**Example**: Including both GDP and industrial production as regressors — they move together.
**Fix**: Check VIF (variance inflation factor). Drop one of the correlated variables. Use PCA for dimensionality reduction.

## 7. Heteroscedasticity
**What**: Residual variance is not constant over time.
**Example**: OLS on financial returns where volatility clusters.
**Fix**: Use GARCH models. Apply White's robust standard errors. Check residual plots for fan-shaped patterns.

## 8. Structural Breaks
**What**: Model parameters change at unknown points in time.
**Example**: A mean-reversion model estimated pre-2008 fails during the crisis because market dynamics changed.
**Fix**: Test for breaks (Chow test, CUSUM). Use rolling windows. Consider Markov switching models. Split estimation periods.

## 9. Small Sample Problems
**What**: Insufficient data for reliable estimation.
**Guidelines**:
- OLS: minimum 30 observations per parameter
- ARIMA: minimum 50 observations
- GARCH: minimum 500 observations
- Markov switching: minimum 200 observations
**Fix**: Reduce model complexity. Use longer sample period. Be cautious about significance claims.

## 10. Overfitting
**What**: Model fits noise in the training data, not the underlying pattern.
**Signals**: High in-sample R² but poor out-of-sample performance. Many insignificant parameters. AIC improves but BIC doesn't.
**Fix**: Use BIC for model selection. Validate out-of-sample. Prefer parsimonious models. Cross-validate.

## Quick Diagnostic Checklist
Before presenting any result, verify:
- [ ] Data is stationary (or appropriately differenced)
- [ ] Sample size is adequate for the model
- [ ] Residuals show no serial correlation (Ljung-Box)
- [ ] Residuals show no heteroscedasticity (or GARCH is used)
- [ ] Parameters are statistically significant
- [ ] Results hold out-of-sample
- [ ] No known structural breaks in the sample
- [ ] Multiple testing correction applied if applicable
