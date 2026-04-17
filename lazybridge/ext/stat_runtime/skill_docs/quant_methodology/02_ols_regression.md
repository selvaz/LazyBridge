# OLS Regression in Quantitative Finance

## When to Use OLS

Ordinary Least Squares regression is appropriate when you need to estimate the linear relationship between a dependent variable and one or more explanatory variables. In quantitative finance, the primary use cases are:

- **Cross-sectional analysis**: Explaining variation in returns across assets at a point in time. Example: regressing stock returns on market beta, size, and value factors (Fama-French).
- **Factor models**: Estimating factor exposures (betas) by regressing an asset's return series on factor return series. Example: `r_i,t = alpha + beta_MKT * MKT_t + beta_SMB * SMB_t + beta_HML * HML_t + epsilon_t`.
- **Simple trend estimation**: Fitting a linear time trend to a series for detrending purposes.
- **Event studies**: Estimating normal returns in an estimation window and computing abnormal returns in an event window.
- **Predictive regressions**: Testing whether a variable (e.g., dividend yield, term spread) predicts future returns.

OLS is the starting point of the modeling toolkit. When it works -- when its assumptions hold -- it is optimal (the Gauss-Markov theorem guarantees minimum-variance unbiased linear estimation). When its assumptions fail, OLS becomes a diagnostic tool that reveals which more sophisticated model is needed.

## The OLS Model

```
Y_t = beta_0 + beta_1 * X_{1,t} + beta_2 * X_{2,t} + ... + beta_k * X_{k,t} + epsilon_t
```

OLS minimizes the sum of squared residuals: `min sum(epsilon_t^2) = min sum(Y_t - X_t * beta)^2`.

The solution is `beta_hat = (X'X)^{-1} X'Y`.

## Key Assumptions (Gauss-Markov + Normality)

### 1. Linearity

The relationship between Y and X is linear in parameters. This does NOT require linearity in variables -- `Y = beta_0 + beta_1 * X + beta_2 * X^2` is linear in parameters and can be estimated by OLS. Transform variables (log, square, interact) as needed to capture nonlinear relationships.

### 2. Exogeneity (E[epsilon | X] = 0)

The error term is uncorrelated with the explanatory variables. This is the most critical and most frequently violated assumption. Violations include:

- **Omitted variable bias**: A variable that affects Y and is correlated with X is excluded. The coefficient on X absorbs the effect of the omitted variable.
- **Simultaneity**: X and Y are determined jointly (e.g., price and quantity).
- **Measurement error in X**: Attenuates the coefficient toward zero.

**In practice**: Exogeneity is a theoretical judgment, not a statistical test. For predictive regressions (e.g., predicting returns from lagged dividend yield), the use of lagged predictors helps, but Stambaugh bias can still distort inference when the predictor is highly persistent.

### 3. Homoscedasticity (Var(epsilon_t) = sigma^2 for all t)

The variance of the error term is constant across observations. In financial data, this assumption almost always fails:

- **Cross-sectional**: Small firms have more volatile returns than large firms. The residual variance is not constant across firm size.
- **Time series**: Volatility clusters (GARCH effects) mean the residual variance changes over time.

**Detection**: White's test, Breusch-Pagan test, or visual inspection of residual-vs-fitted plots.

**Consequence of violation**: OLS coefficients remain unbiased but standard errors are biased (usually downward), making t-statistics too large and p-values too small. You may conclude a variable is significant when it is not.

### 4. No Autocorrelation (Cov(epsilon_t, epsilon_s) = 0 for t != s)

Error terms are uncorrelated across observations. In cross-sectional data, this is usually satisfied. In time-series data, it almost always fails -- financial returns exhibit at least weak serial dependence, and regression residuals in time-series contexts are often autocorrelated.

**Detection**: Durbin-Watson test (for AR(1) autocorrelation), Breusch-Godfrey test (for higher-order autocorrelation), or inspection of the residual ACF.

**Consequence of violation**: Standard errors are biased (downward for positive autocorrelation), inflating t-statistics and producing spurious significance.

### 5. Normality of Errors

For finite-sample inference (exact t-tests, F-tests, confidence intervals), the errors must be normally distributed. Financial return data is famously non-normal: fat tails (excess kurtosis of 3-10 for daily returns) and negative skewness are the rule.

**Detection**: Jarque-Bera test, Shapiro-Wilk test, QQ-plot.

**Consequence of violation**: In large samples (T > 100), the Central Limit Theorem ensures that coefficient estimates are approximately normal regardless of error distribution, so this assumption is less critical than the others. For small samples, non-normality distorts p-values. Fat tails increase the probability of outlier-driven results.

## Diagnostics Checklist

### R-Squared Interpretation

R-squared measures the fraction of variance in Y explained by the model: `R^2 = 1 - SS_res / SS_tot`.

| Context | Typical R^2 | Interpretation |
|---|---|---|
| Cross-sectional stock returns (monthly) | 0.01 - 0.05 | Normal for single-period return prediction |
| Factor model (daily, single stock) | 0.10 - 0.50 | Depends on the asset; market beta alone often gives 0.10-0.30 |
| Factor model (daily, diversified portfolio) | 0.60 - 0.95 | Well-diversified portfolios load heavily on systematic factors |
| Time-series return prediction (daily) | 0.00 - 0.03 | Even 0.5% out-of-sample R^2 is economically meaningful |
| Volatility prediction (GARCH-type) | 0.20 - 0.50 | Against realized volatility |
| Spurious regression (two random walks) | 0.70 - 0.99 | Meaningless; check stationarity |

**Critical rule**: High R-squared does NOT mean the model is good. It can result from spurious regression (non-stationary data), overfitting, or a tautological relationship. Low R-squared does NOT mean the model is useless -- for return prediction, even tiny R-squared values translate to economically significant portfolio improvements.

### Adjusted R-Squared

```
R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - k - 1)
```

Penalizes for additional parameters. Always report adjusted R-squared for models with more than one predictor. If adjusted R-squared drops when a variable is added, that variable is not contributing enough to justify its inclusion.

### F-Test (Overall Significance)

Tests the null hypothesis that ALL slope coefficients are simultaneously zero: `H0: beta_1 = beta_2 = ... = beta_k = 0`.

```
F = (R^2 / k) / ((1 - R^2) / (n - k - 1))
```

A significant F-test (p < 0.05) means at least one predictor has explanatory power. An insignificant F-test means the model as a whole does not explain Y better than the intercept alone.

**Note**: In large samples, the F-test almost always rejects for financial data because even tiny effects are statistically detectable. Statistical significance does not imply economic significance.

### T-Tests on Individual Coefficients

Each coefficient has a t-statistic: `t = beta_hat / SE(beta_hat)`. Under standard assumptions, `t ~ t(n-k-1)`. For n > 100, approximate as standard normal.

| p-value | Evidence Against H0: beta_j = 0 |
|---|---|
| p < 0.01 | Strong |
| 0.01 <= p < 0.05 | Moderate |
| 0.05 <= p < 0.10 | Weak (marginal) |
| p >= 0.10 | Insufficient |

**Always use heteroscedasticity-robust standard errors** (White/HC3) for cross-sectional data and **Newey-West (HAC) standard errors** for time-series data. Standard OLS standard errors are almost never valid for financial data.

### Durbin-Watson Test

Tests for first-order autocorrelation in residuals:

```
DW = sum_{t=2}^{T} (e_t - e_{t-1})^2 / sum_{t=1}^{T} e_t^2
```

- DW ~ 2: No autocorrelation.
- DW < 2: Positive autocorrelation (DW < 1.5 is concerning; DW < 1.0 is strong evidence).
- DW > 2: Negative autocorrelation (DW > 2.5 is concerning).

**The Granger-Newbold rule**: If R^2 > DW, the regression is almost certainly spurious (non-stationary data).

### Jarque-Bera Test

Tests for normality of residuals by checking skewness and kurtosis:

```
JB = (n/6) * (S^2 + (K-3)^2 / 4)
```

where S is skewness and K is kurtosis. Under normality, `JB ~ chi^2(2)`.

For daily financial returns, Jarque-Bera almost always rejects normality (excess kurtosis is typically 3-10). This is expected and does not invalidate the regression for large samples, but it means exact small-sample inference is unreliable and tail-risk estimates based on normality are too optimistic.

## When OLS Fails: Escalation Paths

### Autocorrelated Residuals → ARIMA

If the Durbin-Watson test or Breusch-Godfrey test indicates serial correlation in residuals, the temporal structure is not captured by the regressors alone. Options:

1. Add lagged dependent variables (AR terms) to the regression.
2. Add lagged residuals (MA terms) -- equivalent to ARIMA.
3. Switch to a full ARIMA model if the autocorrelation structure is complex.
4. Use Newey-West standard errors if the goal is inference on the existing coefficients rather than forecasting.

### Heteroscedastic Residuals → GARCH

If White's test or ARCH-LM test rejects homoscedasticity, and especially if the ACF of squared residuals shows significant autocorrelation, the residual variance is time-varying. Options:

1. Use heteroscedasticity-robust (HC) or Newey-West (HAC) standard errors for inference. This corrects standard errors but does not model the variance process.
2. Fit a GARCH model for the residual variance if you need volatility forecasts or risk estimates.
3. Use GLS (Generalized Least Squares) if the form of heteroscedasticity is known.

### Structural Breaks → Markov Regime-Switching

If residuals show patterns suggesting parameter instability (CUSUM test rejects, Chow test rejects, sub-sample parameter estimates differ substantially), the linear relationship may be regime-dependent. Options:

1. Split the sample at the break point and estimate separately.
2. Include dummy variables for known break dates.
3. Fit a Markov regime-switching regression that allows coefficients to differ by regime.

## Common Pitfalls

### 1. High R-Squared Does Not Mean Causation

R-squared measures correlation, not causation. A regression of ice cream sales on drowning deaths will show a significant positive relationship because both are driven by temperature (an omitted variable). In finance, regressing fund returns on subsequently discovered factors does not establish that those factors "drive" returns.

### 2. Omitted Variable Bias

If a relevant variable is excluded and it is correlated with an included variable, the coefficient on the included variable is biased. The bias equals `beta_omitted * Cov(X_included, X_omitted) / Var(X_included)`. The direction of bias depends on the sign of the correlation and the effect of the omitted variable.

**In practice**: There is always an omitted variable. The question is whether the bias is large enough to change conclusions. Report sensitivity to including/excluding plausible control variables.

### 3. Multicollinearity

When predictors are highly correlated (VIF > 10), individual coefficients are estimated imprecisely:

- Standard errors inflate.
- Coefficients become sensitive to which other variables are included.
- Signs may flip unexpectedly.

**Detection**: Compute VIF for each predictor. VIF = 1/(1 - R^2_j), where R^2_j is from regressing X_j on all other predictors.

**Fix**: Drop redundant predictors, use principal components, or use ridge regression.

### 4. Spurious Regression

Regressing non-stationary variables on each other produces meaningless results. This is the single most common econometric error in applied finance. Signs include R^2 > DW, very high R^2 for variables that should be weakly related, and non-stationary residuals.

**Fix**: Difference all variables to achieve stationarity, or test for cointegration.

### 5. Using OLS Standard Errors for Time-Series Data

Standard OLS standard errors assume iid errors. Financial time-series residuals are virtually never iid -- they exhibit autocorrelation and heteroscedasticity. Always use Newey-West (HAC) standard errors for time-series regressions. The Newey-West bandwidth should be proportional to the sample size; a common rule is `int(4*(T/100)^{2/9})`.

## Practical Rules

### Minimum Observations Per Parameter

A common rule of thumb is 10-20 observations per estimated parameter (including the intercept). For a regression with 5 predictors plus an intercept (6 parameters), aim for at least 60-120 observations. With fewer observations:

- Standard errors are large.
- The F-test has low power.
- The model is prone to overfitting.
- Small-sample distributions (t, F) rather than asymptotic approximations should be used.

For financial data, where signal-to-noise ratios are low, err toward the higher end: 30+ observations per parameter for reliable inference.

### When to Add a Constant (Intercept)

**Almost always include a constant.** Excluding the constant forces the regression line through the origin, which:

- Biases all other coefficient estimates if the true intercept is nonzero.
- Makes R-squared meaningless (it can be negative or arbitrarily large with a forced-zero intercept).
- Is only appropriate when theory strongly dictates (e.g., CAPM implies alpha = 0, but you should still estimate alpha and test whether it is zero).

**Exception**: When both Y and all X variables are demeaned (centered), the intercept is zero by construction and can be omitted.

### Interpreting Coefficients

- `beta_j` is the expected change in Y for a one-unit change in X_j, **holding all other variables constant**.
- For log-level models (`ln(Y) = beta_0 + beta_1 * X`): a one-unit change in X is associated with a `beta_1 * 100`% change in Y (approximately, for small beta_1).
- For log-log models (`ln(Y) = beta_0 + beta_1 * ln(X)`): `beta_1` is the elasticity -- a 1% change in X is associated with a `beta_1`% change in Y.
- For standardized coefficients (both Y and X standardized to zero mean, unit variance): `beta_j` measures the change in Y in standard deviations per one standard deviation change in X_j. Use these for comparing the relative importance of predictors measured in different units.

### When to Use Robust Regression

If there are extreme outliers (leverage points) that distort OLS estimates:

- **Winsorize**: Clip extreme values at the 1st and 99th percentiles (or 5th/95th for aggressive trimming). Standard practice for cross-sectional financial data.
- **Robust regression**: Use M-estimators (e.g., Huber) that downweight outliers. Useful when you cannot simply discard extreme observations.
- **Median regression (quantile regression at tau=0.5)**: Minimizes absolute deviations instead of squared deviations. Less sensitive to outliers in Y.

## Summary: OLS Decision Framework

```
Have data with Y and X variables?
├── Are all variables stationary?
│   ├── No → Difference or establish cointegration. Do NOT run OLS on non-stationary levels.
│   └── Yes → Proceed
├── Fit OLS with robust standard errors (HC or Newey-West)
├── Check diagnostics:
│   ├── R^2 > DW? → Likely spurious. Check stationarity again.
│   ├── Durbin-Watson < 1.5? → Autocorrelation. Add AR terms or switch to ARIMA.
│   ├── ARCH-LM rejects? → Heteroscedasticity. Consider GARCH for residual variance.
│   ├── VIF > 10? → Multicollinearity. Drop redundant variables.
│   ├── Jarque-Bera rejects? → Non-normal errors. Acceptable for large samples; use robust SE.
│   └── All diagnostics pass → Report results with appropriate caveats.
└── If OLS is inadequate, escalate to ARIMA (mean dynamics) or GARCH (variance dynamics).
```
