# ARIMA Modeling for Financial Time Series

## What ARIMA Does

ARIMA (Autoregressive Integrated Moving Average) models capture the temporal dependence structure in a single time series. The model exploits two types of serial dependence: autoregressive (AR) dependence on past values and moving average (MA) dependence on past forecast errors. The "Integrated" component handles non-stationarity through differencing.

ARIMA is the natural choice when:

- A single time series exhibits significant autocorrelation (the ACF or PACF has significant lags).
- You need short-horizon point forecasts (1-20 steps ahead) for a series with identifiable serial structure.
- You want a mean model to pair with a GARCH variance model (ARMA-GARCH).
- You need to remove predictable mean dynamics before analyzing residuals for other features (volatility clustering, regime changes).

## Box-Jenkins Methodology

The Box-Jenkins approach to ARIMA modeling consists of three iterative stages:

### Stage 1: Identification

Determine the appropriate model order (p, d, q) by:

1. **Testing for stationarity**: ADF and KPSS tests determine `d` (the differencing order).
2. **Examining ACF and PACF**: The pattern of autocorrelations and partial autocorrelations reveals the AR and MA orders.
3. **Considering candidate models**: Based on ACF/PACF patterns, nominate 2-5 candidate specifications.

### Stage 2: Estimation

Estimate parameters of each candidate model by maximum likelihood (or conditional least squares). Check that:

- The optimizer converged.
- All parameters are statistically significant (p < 0.05).
- AR roots lie outside the unit circle (stationarity condition).
- MA roots lie outside the unit circle (invertibility condition).

### Stage 3: Diagnostic Checking

Verify the model adequacy by examining residuals:

- **Ljung-Box test on residuals**: No remaining autocorrelation (p > 0.05 at lags 10 and 20).
- **ACF of residuals**: No significant spikes.
- **ARCH-LM test**: Check for remaining heteroscedasticity. If significant, consider adding GARCH.
- **Jarque-Bera test**: Check residual normality.

If diagnostics fail, return to Stage 1 and revise the model order.

## ACF/PACF Interpretation for Order Selection

The autocorrelation function (ACF) and partial autocorrelation function (PACF) are the primary tools for identifying ARIMA orders. The PACF at lag k measures the correlation between X_t and X_{t-k} after removing the effects of intermediate lags.

### Pure AR(p) Signatures

An AR(p) process has:
- **ACF**: Decays gradually (exponentially or with damped oscillations). Does NOT cut off sharply.
- **PACF**: Cuts off sharply after lag p. The PACF is significant at lags 1 through p and insignificant afterward.

| Model | ACF Pattern | PACF Pattern |
|---|---|---|
| AR(1), phi > 0 | Exponential decay | Spike at lag 1, then zero |
| AR(1), phi < 0 | Alternating sign decay | Spike at lag 1 (negative), then zero |
| AR(2) | Decay (possibly oscillating) | Spikes at lags 1-2, then zero |

### Pure MA(q) Signatures

An MA(q) process has:
- **ACF**: Cuts off sharply after lag q. The ACF is significant at lags 1 through q and zero afterward.
- **PACF**: Decays gradually. Does NOT cut off sharply.

| Model | ACF Pattern | PACF Pattern |
|---|---|---|
| MA(1), theta > 0 | Spike at lag 1, then zero | Exponential decay (negative) |
| MA(1), theta < 0 | Spike at lag 1 (negative), then zero | Alternating decay |
| MA(2) | Spikes at lags 1-2, then zero | Gradual decay |

### Mixed ARMA(p,q) Signatures

ARMA processes are harder to identify from ACF/PACF because:
- **ACF**: Decays gradually (starting from lag q).
- **PACF**: Decays gradually (starting from lag p).
- Neither function cuts off sharply.

When you see both ACF and PACF decaying without sharp cutoffs, consider ARMA(1,1) as the starting specification. Use information criteria (AIC/BIC) to compare ARMA(1,0), ARMA(0,1), ARMA(1,1), and ARMA(2,1).

### Practical Notes for Financial Data

- **Daily equity returns**: ACF is typically small and often insignificant. If any structure exists, it is usually AR(1) with |phi| < 0.10. Many daily return series are well-modeled as white noise (ARIMA(0,0,0)), with all the action in the conditional variance (GARCH).
- **Weekly or monthly returns**: More likely to show AR(1) structure, especially for less liquid assets or portfolios.
- **Interest rate changes**: Often show AR(1) or AR(2) structure with moderate persistence.
- **Volatility series** (realized volatility, absolute returns): Show strong and persistent autocorrelation. AR models with many lags or fractionally integrated models (ARFIMA) may be needed.

## The (p, d, q) Parameters

### p: Autoregressive Order

The number of lagged values of the series included in the model:

```
X_t = phi_1 * X_{t-1} + phi_2 * X_{t-2} + ... + phi_p * X_{t-p} + epsilon_t
```

- `p = 0`: No AR component. The series does not depend on its own past (beyond what differencing removes).
- `p = 1`: First-order autoregression. The single most common AR specification for financial returns.
- `p = 2`: Second-order. Useful when ACF shows damped oscillation (two complex conjugate AR roots).
- `p >= 3`: Uncommon for financial returns. If needed, consider whether the apparent high-order AR is actually a seasonal pattern or an artifact of a structural break.

**Typical values for financial data**: p = 0 or p = 1 for daily returns. p = 1 to 3 for lower-frequency data. p = 5-20 for realized volatility series (HAR-type models use p at lags 1, 5, 22).

### d: Differencing Order

The number of times the series is differenced to achieve stationarity:

```
d = 0: model the series directly (already stationary)
d = 1: model the first difference Delta X_t = X_t - X_{t-1}
d = 2: model the second difference Delta^2 X_t = Delta X_t - Delta X_{t-1}
```

- `d = 0`: The series is stationary. This is the case for daily financial returns, interest rate changes, and most transformed financial series.
- `d = 1`: The series is I(1). This applies to price levels, cumulative returns, and integrated macro variables. The model for the differenced series (returns) is then ARMA(p,q).
- `d = 2`: Rarely appropriate for financial data. If the first difference is still non-stationary, reconsider the data transformation. Possible exceptions: some nominal economic series with accelerating trends.

**Decision rule**: Use ADF/KPSS tests to determine d. For daily financial returns, d = 0 is almost always correct. For price levels, d = 1 is standard. If you find yourself setting d = 2, stop and reconsider.

### q: Moving Average Order

The number of lagged forecast errors included:

```
X_t = epsilon_t + theta_1 * epsilon_{t-1} + theta_2 * epsilon_{t-2} + ... + theta_q * epsilon_{t-q}
```

- `q = 0`: No MA component. Shocks affect only the current period.
- `q = 1`: First-order MA. Common as part of ARMA(1,1) or ARIMA(0,1,1).
- `q >= 2`: Less common. If the ACF cuts off after lag 2, try MA(2).

**Typical values for financial data**: q = 0 or q = 1. Higher-order MA terms are rarely needed and often indicate over-differencing or model misspecification.

## SARIMAX: Seasonal Extension

### When to Use Seasonal ARIMA

Use SARIMAX (Seasonal ARIMA with eXogenous variables) when:

- The data has a clear seasonal pattern at a known period (s = 12 for monthly, s = 4 for quarterly, s = 5 for weekly with daily data).
- The ACF shows significant spikes at seasonal lags (lag 12, 24 for monthly data).
- The data is macroeconomic or fundamental (GDP, earnings, retail sales) rather than high-frequency financial.

The SARIMAX model is ARIMA(p,d,q)(P,D,Q)[s]:

```
phi(B) * Phi(B^s) * Delta^d * Delta_s^D * X_t = theta(B) * Theta(B^s) * epsilon_t
```

where:
- `(p,d,q)` are the non-seasonal orders.
- `(P,D,Q)` are the seasonal orders.
- `s` is the seasonal period.
- `B` is the backshift operator.

### Common Seasonal Specifications

| Data | Period (s) | Typical Specification |
|---|---|---|
| Monthly macro data | 12 | ARIMA(1,1,1)(1,1,1)[12] |
| Quarterly GDP growth | 4 | ARIMA(1,0,0)(1,1,0)[4] |
| Daily with day-of-week effects | 5 | ARIMA(1,0,1)(1,0,1)[5] |

### When NOT to Use Seasonal ARIMA

- For daily equity returns: there is no strong seasonal pattern at any fixed period. Day-of-week effects are weak and better captured by dummy variables than seasonal differencing.
- For intraday data: apparent periodicity (U-shaped intraday volume) is better handled by deseasonalization than seasonal ARIMA.
- When the "season" is not fixed: if the cycle length varies (business cycles), seasonal ARIMA with a fixed period is inappropriate.

### Exogenous Variables (the X in SARIMAX)

SARIMAX allows inclusion of exogenous regressors:

```
X_t = beta * Z_t + ARIMA_component
```

Use exogenous variables when:
- A known external variable affects the series (e.g., interest rate changes affect bond returns).
- You want to control for deterministic effects (day-of-week dummies, holiday indicators).
- You are building a transfer function model.

**Caution**: Out-of-sample forecasts with exogenous variables require forecasts of the exogenous variables themselves. If you cannot forecast Z_{t+h}, you cannot produce unconditional forecasts from the model.

## Auto Order Selection

### Using AIC/BIC to Compare Candidate Orders

Rather than relying solely on ACF/PACF visual interpretation, fit a grid of candidate models and compare information criteria:

```
Candidate grid (example):
ARIMA(0,0,0), ARIMA(1,0,0), ARIMA(0,0,1), ARIMA(1,0,1),
ARIMA(2,0,0), ARIMA(0,0,2), ARIMA(2,0,1), ARIMA(1,0,2),
ARIMA(2,0,2)
```

For each candidate:
1. Estimate by MLE.
2. Verify convergence and parameter significance.
3. Record AIC and BIC.
4. Select the model with the lowest BIC (for parsimony) or lowest AIC (for prediction).

### Interpreting AIC/BIC Differences

| delta_AIC or delta_BIC | Support for this model |
|---|---|
| 0-2 | Substantial; model is competitive |
| 2-4 | Some support |
| 4-7 | Weak support |
| > 10 | Essentially no support |

If the top 2-3 models have delta < 2, they are effectively equivalent. Choose the most parsimonious.

### Automated Selection Algorithms

The `auto_arima` approach (analogous to R's `auto.arima`):
1. Determine d using unit root tests (ADF, KPSS).
2. Search over a grid of (p,q) values, typically p in {0,...,5} and q in {0,...,5}.
3. Use stepwise search starting from ARIMA(0,d,0), ARIMA(1,d,0), ARIMA(0,d,1) and expanding.
4. Select by AIC or BIC.

**Practical rule**: Trust automated selection as a starting point, but always verify with diagnostics. Automated methods can select models with insignificant parameters or poor residual properties.

## Differencing

### First Difference for Unit Root

If ADF/KPSS indicate a unit root (I(1) process):

```
Delta X_t = X_t - X_{t-1}
```

In the ARIMA(p,1,q) framework, the model is estimated on the differenced series internally.

**For financial prices**: Always difference (or equivalently, use log returns). Never model price levels with ARMA.

### Seasonal Differencing

For seasonal non-stationarity (seasonal ACF spikes that do not decay):

```
Delta_s X_t = X_t - X_{t-s}
```

Seasonal differencing can be combined with regular differencing:

```
Delta Delta_s X_t = (X_t - X_{t-s}) - (X_{t-1} - X_{t-1-s})
```

**Decision rule**: If the ACF at the seasonal lag (12 for monthly) is very large (> 0.7) and does not decay, seasonal differencing is needed (D=1). After seasonal differencing, re-examine the ACF/PACF for the non-seasonal orders.

### Detecting Over-Differencing

If the first difference of a stationary series is computed, the resulting series has:
- Artificially induced negative autocorrelation at lag 1 (rho_1 ~ -0.5).
- An MA(1) signature in the ACF (spike at lag 1 only).

**Rule**: If the ACF of the differenced series shows rho_1 near -0.5, the original series was likely already stationary. Remove the differencing (set d = 0).

## Forecasting with ARIMA

### Point Forecasts

ARIMA produces optimal (minimum MSE) linear forecasts conditional on the estimated model. For an ARMA(1,1) with estimated parameters:

```
X_hat_{T+1} = phi * X_T + theta * epsilon_T
X_hat_{T+2} = phi * X_hat_{T+1}
X_hat_{T+h} = phi * X_hat_{T+h-1}   (for h > q, the MA term drops out)
```

Multi-step forecasts converge to the unconditional mean as the horizon increases. For a stationary ARMA process, the forecast approaches `mu = E[X_t]` exponentially fast.

### Confidence Intervals

Forecast uncertainty grows with the horizon:

```
Var(e_{T+h}) = sigma^2 * (1 + psi_1^2 + psi_2^2 + ... + psi_{h-1}^2)
```

where `psi_j` are the MA(infinity) representation coefficients.

- **1-step-ahead**: Uncertainty is just `sigma^2` (the innovation variance).
- **Multi-step**: Uncertainty accumulates. For an AR(1) with phi=0.5, the 10-step-ahead forecast variance is ~1.33 * sigma^2. For phi=0.9, it is ~6.5 * sigma^2.
- **Long horizon**: Forecast variance converges to the unconditional variance of the process. The forecast is essentially the unconditional mean, and the confidence interval spans the full distribution.

**Practical implication**: ARIMA forecasts are useful for short horizons (1-5 steps for daily data, 1-4 quarters for quarterly data). Beyond that, the confidence intervals are so wide that the forecast is uninformative.

### Forecast Evaluation

Always evaluate forecasts out of sample:
- Hold out the last 20-30% of observations.
- Produce rolling or expanding-window forecasts.
- Compare against a naive benchmark (random walk for prices; historical mean for returns).
- Compute RMSE, MAE, and directional accuracy.
- Report the out-of-sample R-squared: `R^2_OOS = 1 - sum(e_t^2) / sum((Y_t - Y_bar)^2)`. Negative values mean the model is worse than the historical mean.

## When ARIMA Is Insufficient

### Volatility Clustering in Residuals → Add GARCH

If the ARIMA model's residuals exhibit:
- Significant autocorrelation in squared residuals (Ljung-Box on e_t^2 rejects).
- Significant ARCH-LM test.
- Visual volatility clustering (periods of large residuals followed by large residuals).

Then the mean model (ARIMA) is adequate, but the variance process needs modeling. Fit an ARMA(p,q)-GARCH(1,1) model where:
- The ARMA component models the conditional mean.
- The GARCH component models the conditional variance.

### Regime Changes → Markov Switching

If the ARIMA residuals show:
- Sub-period parameter instability (rolling estimates of phi vary substantially).
- CUSUM test rejection.
- Evidence of distinct behavioral periods in the residual plot.

Then a single-regime ARIMA model is averaging over regimes. Consider Markov-switching ARIMA where the AR/MA parameters (or at least the mean and variance) differ by regime.

### Nonlinear Dependence → Threshold or Nonlinear Models

If the ARIMA residuals show:
- Significant BDS test (tests for nonlinear serial dependence).
- Asymmetric behavior (negative returns have different predictability than positive returns).

Consider threshold AR (TAR/SETAR) models or neural network approaches. However, nonlinear models require substantially more data and are prone to overfitting.

## Practical Rules for Financial ARIMA Modeling

### Rule 1: Daily Financial Returns Rarely Need d > 0

Log returns of stocks, indices, currencies, and commodities are almost always stationary. If ADF does not reject for daily returns, check for data errors (you may be inadvertently using price levels). Set d = 0 and focus on the ARMA specification.

### Rule 2: Keep Orders Low

Do not exceed p = 5 or q = 5 for the non-seasonal component. If the optimal model appears to be ARIMA(6,0,3), something is wrong:
- The series may have structural breaks that create apparent high-order dependence.
- The series may have a seasonal pattern that should be handled by SARIMAX or dummies.
- You may be fitting noise.

**Standard range**: p in {0, 1, 2, 3} and q in {0, 1, 2} covers the vast majority of financial applications.

### Rule 3: Prefer Parsimony When Models Are Close

If ARIMA(1,0,0) has AIC = -5234.2 and ARIMA(2,0,1) has AIC = -5236.1, the improvement of 1.9 AIC points is within the "essentially equivalent" range (delta < 2). Choose ARIMA(1,0,0).

### Rule 4: Check Parameter Significance

After selecting a model by AIC/BIC, verify that all parameters are significant at the 5% level. If a parameter is insignificant, consider whether a simpler nested model would suffice. An ARIMA(2,0,1) where phi_2 is insignificant may be outperformed by ARIMA(1,0,1).

### Rule 5: Always Check Residuals

No model selection is complete without residual diagnostics. The Ljung-Box test at lags 10 and 20 is the minimum. If residuals show structure, the model is misspecified regardless of what AIC/BIC say.

### Rule 6: Consider the Purpose

- For **forecasting**: AIC-selected models may be preferred (better prediction).
- For **inference** (testing whether autocorrelation exists): BIC-selected models are more reliable (more parsimonious, lower false positive rate).
- For **mean model within ARMA-GARCH**: The mean model order matters less than the variance model. A simple AR(1) or even a constant mean is often adequate; the GARCH component does the heavy lifting for risk applications.

### Rule 7: Be Skeptical of Predictability in Efficient Markets

For daily equity returns, statistically significant AR(1) coefficients with |phi| < 0.05 are common but economically negligible (R-squared < 0.25%). Before building a forecasting model, ask whether the predictability is large enough to be exploitable after transaction costs. If the out-of-sample R-squared is below 0.5%, the ARIMA forecast is unlikely to generate economic value.

## Parameter Reporting Conventions

When reporting ARIMA results, include:

- Model specification: ARIMA(p,d,q) or SARIMAX(p,d,q)(P,D,Q)[s]
- All parameter estimates with standard errors and p-values
- AIC, BIC, log-likelihood
- Ljung-Box p-values at lags 10 and 20 for residuals and squared residuals
- ARCH-LM test p-value (to assess need for GARCH)
- Sample period, frequency, and number of observations
- Any data transformations (log, difference, seasonal adjustment)
- Out-of-sample forecast performance if forecasting is the goal
- Comparison with naive benchmark (random walk or historical mean)
