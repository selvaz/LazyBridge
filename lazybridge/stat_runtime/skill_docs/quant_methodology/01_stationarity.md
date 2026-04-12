# Stationarity Testing and Transformation

## What Stationarity Means

A time series is **stationary** if its statistical properties do not depend on the time at which the series is observed. Informally, a stationary series has no trend, no changing variance, and no periodic fluctuations that are not already accounted for. The series fluctuates around a constant mean with constant variance and a fixed autocorrelation structure.

Stationarity is not a technicality -- it is the foundational requirement for valid inference in time-series econometrics. When stationarity fails, standard regression and time-series tools produce unreliable results: inflated R-squared values, invalid t-statistics, and spurious apparent relationships between unrelated variables.

## Why Stationarity Matters for Every Model Family

- **OLS regression**: If the dependent variable or regressors are non-stationary, the regression may be spurious. R-squared can exceed 0.90 even when the variables are completely unrelated (Granger and Newbold, 1974). T-statistics do not follow their usual distributions. Standard errors are biased.
- **ARIMA**: The AR and MA components assume stationarity. The `d` parameter in ARIMA(p,d,q) explicitly handles non-stationarity by differencing, but you must correctly identify the order of integration.
- **GARCH**: Requires the return series to be covariance-stationary. Fitting GARCH to price levels (which are I(1)) produces nonsensical persistence parameters near or above 1.0 and meaningless volatility forecasts.
- **Markov regime-switching**: Assumes stationarity within each regime and stationarity of the overall switching process. If the data has a deterministic trend overlaid on regime switching, the model will misidentify regimes.

## Types of Stationarity

### Strict (Strong) Stationarity

The joint distribution of `(X_{t1}, X_{t2}, ..., X_{tk})` is identical to the joint distribution of `(X_{t1+h}, X_{t2+h}, ..., X_{tk+h})` for all choices of time points and all shifts `h`. This means every aspect of the distribution -- including all moments, tail behavior, and dependence structure -- is time-invariant.

Strict stationarity is rarely assumed in practice because it is both difficult to test and unnecessarily strong for most applications.

### Weak (Covariance/Second-Order) Stationarity

A less demanding but practically sufficient condition. A series is weakly stationary if:

1. **Constant mean**: `E[X_t] = mu` for all t.
2. **Constant variance**: `Var(X_t) = sigma^2` for all t.
3. **Autocovariance depends only on lag**: `Cov(X_t, X_{t+h}) = gamma(h)` for all t.

Weak stationarity is the working definition in financial econometrics. When we say "stationary" without qualification, we mean weakly stationary.

**Key distinction**: A weakly stationary series can have time-varying higher moments. GARCH processes are weakly stationary (constant unconditional variance) but have time-varying conditional variance. This is fine -- GARCH models the conditional variance while maintaining unconditional stationarity.

## Unit Roots and Non-Stationarity

### What a Unit Root Is

A unit root is a feature of the autoregressive representation of a time series that causes non-stationarity. Consider the AR(1) process:

```
X_t = phi * X_{t-1} + epsilon_t
```

- If `|phi| < 1`: the process is stationary. Shocks decay geometrically; the series reverts to its mean.
- If `phi = 1`: the process has a **unit root**. Shocks have permanent effects; the series is a random walk. This is an I(1) process.
- If `|phi| > 1`: the process is explosive and not encountered in financial data.

### Random Walk

The simplest unit root process:

```
X_t = X_{t-1} + epsilon_t
```

A random walk has no tendency to return to any level. Its variance grows linearly with time: `Var(X_t) = t * sigma^2_epsilon`. This is why asset prices (which approximate random walks) have volatility that grows with the square root of the holding period.

### I(1) Processes

A series that is non-stationary but becomes stationary after first differencing is said to be integrated of order 1, or I(1). Most asset prices are I(1): the price level is non-stationary, but the first difference (or log return) is stationary.

```
Price:   P_t is I(1)     (non-stationary)
Return:  r_t = P_t - P_{t-1} is I(0)   (stationary)
```

Higher orders of integration (I(2), I(3)) are rare in financial data. If a series requires two or more differences to achieve stationarity, reconsider whether the transformation is appropriate.

### Random Walk with Drift

```
X_t = mu + X_{t-1} + epsilon_t
```

The drift term `mu` introduces a deterministic trend. Stock prices often follow this pattern: a random walk with a positive drift (expected positive return). The first difference removes both the unit root and the drift: `Delta X_t = mu + epsilon_t`, which is stationary (constant mean mu).

## The ADF Test (Augmented Dickey-Fuller)

### Setup

The ADF test is the most widely used unit root test. It tests the null hypothesis:

```
H0: The series has a unit root (non-stationary)
H1: The series is stationary
```

The test regression is:

```
Delta X_t = alpha + beta*t + gamma*X_{t-1} + sum_{i=1}^{p} delta_i * Delta X_{t-i} + epsilon_t
```

where `alpha` is a constant, `beta*t` is a time trend (optional), and the lagged differences account for serial correlation. The test statistic is the t-statistic on `gamma`. Under H0, `gamma = 0` (unit root); under H1, `gamma < 0`.

### Critical Values

The ADF test statistic does NOT follow a standard t-distribution. It follows the Dickey-Fuller distribution, which has heavier left tails. Critical values depend on the deterministic specification:

| Specification | 1% critical value | 5% critical value | 10% critical value |
|---|---|---|---|
| No constant, no trend | -2.58 | -1.95 | -1.62 |
| Constant, no trend | -3.43 | -2.86 | -2.57 |
| Constant + trend | -3.96 | -3.41 | -3.13 |

These are for T=500; values vary slightly with sample size.

### Interpretation

- **Reject H0** (test statistic more negative than critical value, or p-value < 0.05): Evidence that the series is stationary. Proceed with stationary-series models.
- **Fail to reject H0** (test statistic less negative than critical value, or p-value > 0.05): Cannot rule out a unit root. The series may be non-stationary. Consider differencing.

### Practical P-Value Thresholds

| p-value | Interpretation | Action |
|---|---|---|
| p < 0.01 | Strong evidence against unit root | Treat as stationary |
| 0.01 <= p < 0.05 | Moderate evidence against unit root | Treat as stationary (with caution) |
| 0.05 <= p < 0.10 | Weak evidence; borderline | Run KPSS for confirmation |
| p >= 0.10 | Insufficient evidence to reject unit root | Likely non-stationary; difference the series |

### Lag Length Selection for ADF

The number of augmentation lags `p` matters. Too few lags leave serial correlation in the residuals, distorting the test. Too many lags reduce power. Selection methods:

- **AIC/BIC minimization**: Fit the test regression with varying `p` (typically 1 to int(12*(T/100)^{1/4})) and choose the `p` that minimizes AIC or BIC.
- **General-to-specific**: Start with a large `p`, remove insignificant lags from the top until the last lag is significant.
- **Default**: For daily financial data with T > 1000, starting with p = int(12*(T/100)^{1/4}) is standard. For T=2500, this gives p ~ 18.

## The KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)

### Setup

KPSS reverses the null and alternative hypotheses relative to ADF:

```
H0: The series is stationary (possibly around a deterministic trend)
H1: The series has a unit root (non-stationary)
```

The test statistic is based on the partial sums of residuals from a regression of the series on a constant (level stationarity) or a constant plus trend (trend stationarity).

### Critical Values

| Specification | 1% critical value | 5% critical value | 10% critical value |
|---|---|---|---|
| Level stationarity | 0.739 | 0.463 | 0.347 |
| Trend stationarity | 0.216 | 0.146 | 0.119 |

**Note**: Unlike ADF, KPSS rejects when the test statistic is LARGE (exceeds the critical value).

### Interpretation

- **Fail to reject H0** (test statistic below critical value): Consistent with stationarity.
- **Reject H0** (test statistic above critical value): Evidence of non-stationarity.

### Bandwidth Selection for KPSS

The KPSS test requires a bandwidth parameter for the long-run variance estimator. The Newey-West automatic bandwidth selection (using the Bartlett kernel) is standard. Using too small a bandwidth overstates the test statistic (too many rejections); too large a bandwidth understates it (too few rejections).

## Why Use Both ADF and KPSS Together

Neither test alone is definitive. ADF has low power against near-unit-root alternatives (a series with phi = 0.98 is stationary but ADF may fail to reject the unit root null). KPSS has size distortions in small samples. Using both tests together provides a more reliable classification.

## The 4-Case Interpretation Matrix

| ADF Result | KPSS Result | Conclusion | Action |
|---|---|---|---|
| Rejects (p < 0.05) | Does not reject | **Stationary** | Proceed with stationary-series models |
| Does not reject (p > 0.05) | Rejects | **Non-stationary (unit root)** | Difference the series; use d=1 in ARIMA |
| Rejects | Rejects | **Trend-stationary** | Detrend (remove deterministic trend), then model the residuals |
| Does not reject | Does not reject | **Inconclusive** | Increase sample size; try with/without trend; consider structural breaks |

### Common Outcomes for Financial Data

- **Log returns of equity indices**: Both tests agree on stationarity (ADF rejects, KPSS does not). This is the typical case.
- **Price levels**: Both tests agree on non-stationarity (ADF does not reject, KPSS rejects). Expected for any asset price.
- **Interest rates**: Often inconclusive or near-unit-root. Short rates may be I(1) in some periods and stationary in others.
- **Volatility measures (VIX, realized vol)**: Usually stationary but highly persistent. ADF may have low power; KPSS usually does not reject.

## How to Fix Non-Stationarity

### First Differencing

The most common transformation. If `X_t` is I(1), then `Delta X_t = X_t - X_{t-1}` is I(0) (stationary).

For prices to returns:
```
r_t = ln(P_t) - ln(P_{t-1})   (log returns, preferred)
r_t = (P_t - P_{t-1}) / P_{t-1}   (simple returns)
```

Log returns are preferred because they are additive over time and approximately equal to simple returns for small magnitudes.

**Caution**: Do not over-difference. If the series is already stationary, differencing introduces unnecessary negative autocorrelation at lag 1 (the MA(1) signature of over-differencing). Check: if the ACF of the differenced series has a large negative spike at lag 1 (rho_1 around -0.5), you may have over-differenced.

### Log Transformation

Applying a log transform before differencing is standard for price data. The log transform:

- Stabilizes variance when variance is proportional to the level (common for prices).
- Converts multiplicative relationships to additive ones.
- Makes the differenced series interpretable as continuously compounded returns.

### Detrending

If the series is trend-stationary (deterministic trend + stationary fluctuations around the trend), remove the trend by regression:

```
X_t = alpha + beta*t + u_t
```

Use the residuals `u_t` for subsequent modeling. This is appropriate when both ADF and KPSS reject (the trend-stationary case).

**Warning**: Detrending a unit root process does NOT make it stationary. If the non-stationarity is stochastic (unit root) rather than deterministic (trend), you must difference, not detrend.

### Seasonal Differencing

For data with seasonal patterns (monthly, quarterly), a seasonal difference may be needed:

```
Delta_s X_t = X_t - X_{t-s}
```

where s is the seasonal period (12 for monthly, 4 for quarterly). Seasonal differencing addresses seasonal unit roots (seasonal non-stationarity).

## Implications Per Model Family

### OLS Regression

OLS requires all variables (dependent and independent) to be stationary, or cointegrated if I(1). Without stationarity or cointegration:

- Regressions are spurious: inflated R-squared, invalid inference.
- The Granger-Newbold rule of thumb: if `R^2 > Durbin-Watson statistic`, the regression is almost certainly spurious.

**What to do**: Difference all variables to achieve stationarity, or establish cointegration and use an error correction model (ECM). For standard cross-sectional or panel analysis, stationarity is typically ensured by using returns rather than price levels.

### ARIMA

ARIMA handles non-stationarity explicitly through the `d` parameter:

- `d = 0`: The series is stationary; fit ARMA(p,q).
- `d = 1`: The series is I(1); the model internally differences once.
- `d = 2`: Rarely needed for financial data. If d=2 seems necessary, reconsider the data transformation.

**Key point**: The ADF/KPSS tests determine whether `d = 0` or `d = 1` is appropriate. For daily financial returns, d = 0 is almost always correct (returns are stationary). For prices or cumulative values, d = 1 is standard.

### GARCH

GARCH models assume the mean process (returns) is stationary. The GARCH variance process itself has a stationarity condition: `alpha + beta < 1` for GARCH(1,1).

**What to do**: Always convert prices to log returns before fitting GARCH. Verify stationarity of returns with ADF/KPSS. If `alpha + beta >= 1` in the estimated model, suspect:

- Non-stationary input data (most common error: accidentally fitting to price levels).
- Structural breaks in the sample.
- The need for an IGARCH specification (integrated GARCH, where alpha + beta = 1 exactly).

### Markov Regime-Switching

Regime-switching models assume stationarity **within** each regime and stationarity of the Markov chain governing regime transitions. The overall series may appear non-stationary (because it switches between regimes with different means and variances), but each individual regime should be stationary.

**What to do**: Work with returns, not prices. If the data has a strong deterministic trend in addition to regime switching, detrend before fitting. If the transition probabilities themselves appear to change over time (e.g., bear markets became more frequent after a policy change), a standard Markov model is misspecified.

## Practical Decision Rules

1. **Always test stationarity before fitting any model.** No exceptions.
2. **Use both ADF and KPSS.** Relying on a single test is insufficient.
3. **For financial returns (daily, weekly, monthly)**: Expect stationarity. If ADF fails to reject, check for data errors (accidentally using prices instead of returns).
4. **For price levels**: Expect I(1). Do not model prices directly with ARMA or GARCH. Convert to returns.
5. **If inconclusive** (both tests fail to reject): Increase the sample size if possible, check for structural breaks, or proceed with caution using both differenced and undifferenced specifications and compare results.
6. **Do not over-difference.** If the series is already stationary, differencing introduces artificial negative autocorrelation. Check the ACF of the differenced series for a large negative spike at lag 1.
7. **For very persistent series** (e.g., volatility, interest rates): The series may be stationary but with a near-unit root. ADF may lack power. In these cases, KPSS not rejecting is informative. Consider fractional integration models as an alternative.
8. **Document the stationarity test results** in every report. State the test used, the test statistic, the p-value, and the conclusion.
