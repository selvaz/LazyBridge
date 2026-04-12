# Common Statistical Pitfalls in Quantitative Finance

## Overview

This document catalogs the most frequent and consequential statistical errors encountered in quantitative finance analysis. Each pitfall is described with its mechanism, how to detect it, and how to avoid or correct it. These are not theoretical curiosities -- they routinely invalidate published research and production models.

## 1. Spurious Regression

### What It Is

When two non-stationary (integrated) time series are regressed on each other, conventional regression statistics (R-squared, t-statistics) are unreliable and often indicate a strong relationship even when none exists. Granger and Newbold (1974) demonstrated that regressing two independent random walks produces significant t-statistics over 70% of the time at the 5% level.

### How to Detect

- **Symptom**: High R-squared (0.70+) with very low Durbin-Watson statistic (< 0.50). The "Granger-Newbold rule of thumb": if R^2 > DW, suspect spurious regression.
- **Test for unit roots**: Run ADF and KPSS tests on all variables. If variables are I(1), OLS in levels is likely spurious unless the variables are cointegrated.
- **Residual stationarity**: If the regression residuals are non-stationary (ADF fails to reject), the regression is almost certainly spurious.

### How to Fix

1. **Difference the data**: Regress first differences (returns) on first differences. This eliminates the unit root but loses long-run information.
2. **Cointegration testing**: If you believe a long-run equilibrium relationship exists, test for cointegration (Engle-Granger two-step, Johansen trace test). If cointegrated, use an error correction model (ECM).
3. **Never regress price levels on price levels** without first establishing cointegration. Regressing S&P 500 price on GDP level will produce a beautiful fit and meaningless inference.

### Concrete Example

Regressing the S&P 500 index level on the number of smartphones sold globally (both trending upward from 2007-2024) will yield R^2 > 0.90 and a highly significant coefficient. This is entirely spurious -- both series are driven by time trends, not by a causal relationship.

## 2. Data Snooping and P-Hacking

### What It Is

Data snooping occurs when the same dataset is used repeatedly to search for statistically significant patterns. If you test 100 independent hypotheses at the 5% level, you expect 5 false positives purely by chance. Selecting and reporting only the significant results without accounting for the multiple testing is p-hacking.

### Forms in Quant Finance

- **Strategy mining**: Testing hundreds of trading rules on the same historical data and reporting only the profitable ones.
- **Variable selection by significance**: Trying 30 predictors in a return regression and keeping only the 3 with p < 0.05.
- **Specification search**: Trying ARIMA(1,1,0), ARIMA(2,1,0), ARIMA(1,1,1), ..., and reporting only the one with the best fit.
- **Lookback period selection**: Testing strategies over 1990-2020, 1995-2020, 2000-2020, and choosing the period that produces the best result.

### How to Detect

- **Too-good-to-be-true results**: Daily Sharpe ratios above 3.0 or return predictability R^2 above 5% from linear models warrant extreme skepticism.
- **Many specifications tested**: If the report mentions "extensive robustness checks" but only shows favorable results, data snooping is likely.
- **No pre-registration**: If the hypothesis was not specified before looking at the data, the p-values are not valid.

### How to Fix

1. **Out-of-sample testing**: Always reserve a holdout period. Results that do not survive out-of-sample are likely spurious.
2. **Multiple testing correction**: Apply Bonferroni correction (divide alpha by number of tests) or the Benjamini-Hochberg procedure (controls false discovery rate).
3. **Bonferroni example**: If you tested 20 specifications, the individual significance level should be 0.05/20 = 0.0025, not 0.05.
4. **Report all tested specifications**: Transparency about the search process allows readers to calibrate their confidence.
5. **Pre-specify the analysis plan**: Decide on the model specification before fitting. Exploratory analysis is fine, but confirmatory claims require a pre-specified test.

### The Harvey, Liu, and Zhu (2016) Threshold

For factor discovery in asset pricing, Harvey et al. argue that the conventional t-statistic threshold of 1.96 is far too low given the collective data mining in the literature. They propose a threshold of approximately 3.0 (corresponding to p < 0.003) to account for the hundreds of factors that have been tested. When evaluating new factors or predictors, apply this higher bar.

## 3. Survivorship Bias

### What It Is

Analyzing only entities (stocks, funds, strategies) that survived to the present, ignoring those that failed, were delisted, or ceased to exist. This systematically overstates returns and understates risk.

### Magnitude

- **Mutual funds**: Survivorship bias in US mutual fund returns is approximately 0.5-1.5% per year (Elton, Gruber, Blake, 1996). Funds that perform poorly are merged or liquidated, so the surviving sample looks artificially good.
- **Stocks**: Using a current stock universe to backtest a strategy ignores delisted companies. A stock that went to zero in 2015 does not appear in a 2024 database snapshot. Ignoring delistings can overstate annual returns by 1-2% for small-cap universes.
- **Hedge funds**: Survivorship bias in hedge fund databases is estimated at 2-4% per year.

### How to Detect

- **Data source inquiry**: Does the database include dead/delisted entities? CRSP includes delisting returns; many commercial databases do not.
- **Declining observation count**: If earlier periods have far fewer observations than later periods (even after controlling for market growth), the historical data may exclude failures.
- **Unrealistic backtest returns**: If a backtest of a simple strategy (e.g., small-cap value) shows returns 2-3% higher than published academic results, survivorship bias is likely the explanation.

### How to Fix

1. **Use survivorship-bias-free databases**: CRSP (stocks), Morningstar (includes dead funds), BarclayHedge (includes dead hedge funds).
2. **Include delisting returns**: When a stock is delisted, assign the delisting return (which is often -30% to -100% for performance-related delistings).
3. **Point-in-time data**: Use constituent lists as of each historical date, not the current constituent list.

## 4. Look-Ahead Bias

### What It Is

Using information that was not available at the time a decision would have been made. This is the most insidious bias in backtesting because it is easy to introduce accidentally and difficult to detect.

### Common Sources

- **Using revised data**: GDP, employment, and other macro variables are revised months after initial release. Using the final revised value in a historical backtest creates look-ahead bias. Always use "vintage" or "real-time" data.
- **Future index constituents**: Backtesting the "S&P 500" using today's constituent list. In 2005, the S&P 500 had different members than it does today. A company added in 2015 should not appear in the 2005 portfolio.
- **Alignment errors**: Using daily closing prices with data that is released after market close (e.g., earnings announcements released at 4:30 PM used as if available at close).
- **Full-sample estimation**: Computing a z-score using the full-sample mean and standard deviation, then using it for signals at each point in time. The mean and std should be computed using only data available up to that point.
- **Feature engineering leaks**: Computing a 20-day moving average that includes the current day's data when generating a signal for the current day's trading decision.

### How to Detect

- **Suspiciously good timing**: If a backtest consistently enters positions right before large moves, look-ahead bias is likely.
- **Performance degrades dramatically out-of-sample**: In-sample Sharpe of 3.0, out-of-sample Sharpe of 0.3 often indicates look-ahead bias (or overfitting).
- **Audit the data pipeline**: For every feature, verify: "Would this value have been known at the time the model uses it?" This is tedious but essential.

### How to Fix

1. **Point-in-time databases**: Use databases that record the value as it was known at each historical date (e.g., Compustat Point-in-Time).
2. **Lag all inputs by one period**: If trading at today's close, features must use only data available before today's close.
3. **Walk-forward methodology**: Re-estimate models at each step using only past data.
4. **Code review**: Have someone else review the backtesting code specifically for look-ahead issues.

## 5. Non-Stationarity Traps

### Beyond Unit Roots

Non-stationarity is not just about unit roots. Several forms create problems:

#### Structural Breaks

The data-generating process changes at an unknown point. A model estimated on pre-2008 data may be useless for post-2008 data if a structural break occurred.

- **Detection**: Chow test (known breakpoint), Bai-Perron test (unknown breakpoints), CUSUM test (recursive residuals).
- **Impact**: Parameter estimates are averages across regimes and may not represent any actual regime.
- **Fix**: Estimate on sub-samples, use regime-switching models, or include dummy variables for known breaks.

#### Time-Varying Parameters

Parameters drift slowly over time rather than shifting abruptly.

- **Detection**: Estimate on rolling windows and check for parameter drift. Plot coefficients over time.
- **Impact**: Full-sample estimates may be misleading for current conditions.
- **Fix**: Use rolling-window estimation or time-varying parameter models (Kalman filter, state-space models).

#### Changing Distributions

The unconditional distribution of returns changes over time (e.g., lower volatility regimes in the 1990s vs higher in the 2000s).

- **Impact**: Tail risk estimates based on the full sample may not reflect current risk.
- **Fix**: Use exponentially weighted estimation or recent sub-samples for distribution fitting.

## 6. Multicollinearity

### What It Is

When two or more predictors in a regression are highly correlated, individual coefficient estimates become unstable (high standard errors) even if the overall model fits well. The model cannot distinguish the individual effects.

### Symptoms

- **Large standard errors with high overall R-squared**: Individual coefficients are insignificant, but the F-test is significant.
- **Sign changes**: Adding or removing a correlated variable flips the sign of another coefficient.
- **VIF > 10**: Variance Inflation Factor above 10 indicates problematic multicollinearity. VIF above 5 warrants attention.

### Common Occurrences in Finance

- Including both the 10-year Treasury yield and the 30-year Treasury yield as predictors (correlation > 0.95).
- Including both market cap and total assets (strongly correlated for most firms).
- Including multiple momentum measures (1-month, 3-month, 6-month returns) that are highly correlated.
- Including both VIX and realized volatility.

### How to Fix

1. **Drop redundant variables**: Keep the variable with the strongest theoretical justification.
2. **Principal Component Analysis**: Replace correlated predictors with their principal components.
3. **Ridge regression**: L2 regularization stabilizes coefficient estimates in the presence of multicollinearity.
4. **Combine variables**: Average or otherwise combine highly correlated predictors into a single measure.

### When Multicollinearity Does NOT Matter

If your goal is **prediction** rather than **inference**, multicollinearity is less problematic. The model will still predict well; you just cannot interpret individual coefficients. However, predictions may be unstable if the correlation structure changes out of sample.

## 7. Heteroscedasticity

### What It Is

Non-constant variance of the error term across observations. In financial time series, this manifests as volatility clustering (addressed by GARCH models). In cross-sectional regressions, it manifests as different error variances for different firms or asset sizes.

### Consequences

- OLS coefficient estimates remain **unbiased** but are **inefficient** (not minimum variance).
- Standard errors are **biased**, typically downward, making t-statistics too large and p-values too small.
- **Inference is invalid**: You may conclude a variable is significant when it is not.

### Detection

- **White's test**: General test for heteroscedasticity. Regress squared residuals on fitted values and their squares. Significant F-test indicates heteroscedasticity.
- **Breusch-Pagan test**: Similar but regresses squared residuals on the original regressors.
- **Visual inspection**: Plot residuals vs fitted values. A fan or funnel shape indicates heteroscedasticity.
- **In time series**: ARCH-LM test detects conditional heteroscedasticity.

### How to Fix

1. **Heteroscedasticity-consistent standard errors (HC)**: Use White (HC0), HC1, HC2, or HC3 standard errors. HC3 is preferred for small samples. This does not change the coefficients but corrects the standard errors and p-values.
2. **Newey-West standard errors**: For time series with both heteroscedasticity and autocorrelation (HAC standard errors). Always use these for time-series regressions.
3. **GLS/WLS**: If the form of heteroscedasticity is known (e.g., variance proportional to firm size), weighted least squares is efficient.
4. **GARCH modeling**: For conditional heteroscedasticity in time series, model the variance process explicitly.

**Critical rule**: In any time-series regression, always report Newey-West (HAC) standard errors, not OLS standard errors. OLS standard errors are almost never valid for financial time series.

## 8. Structural Breaks

### Why They Matter

Financial markets undergo regime changes: policy shifts, regulatory changes, technological disruptions, crises. A model estimated across a structural break produces parameter estimates that are averages of the pre-break and post-break values, potentially representing neither regime well.

### Examples

- **2008 Financial Crisis**: Volatility dynamics, correlations, and mean returns all changed substantially.
- **Zero Interest Rate Policy (2009-2015, 2020-2022)**: Changed the relationship between interest rates and asset prices.
- **COVID-19 (2020)**: Introduced unprecedented return patterns and correlation breakdowns.
- **Decimalization (2001)**: Changed market microstructure and bid-ask spreads.

### Detection Methods

| Test | Use Case | Output |
|---|---|---|
| Chow test | Known breakpoint | F-statistic for parameter change |
| Bai-Perron | Unknown breakpoints | Estimated break dates and confidence intervals |
| CUSUM | Recursive stability | Plot of cumulative sum of recursive residuals |
| CUSUM-sq | Variance stability | Detects variance changes |
| Quandt-Andrews | Unknown single break | Supremum F-statistic across possible breaks |

### How to Handle

1. **Sub-sample estimation**: Estimate separately on pre-break and post-break periods. Report both.
2. **Dummy variables**: Include a dummy for the break period if the break is known.
3. **Regime-switching models**: Let the model identify breaks endogenously.
4. **Rolling windows**: Use a rolling estimation window that adapts to the most recent regime.
5. **Forecast with most recent data**: For forecasting, prioritize recent data that reflects the current regime over a longer sample that includes outdated regimes.

### The Sample Length Tradeoff

- **Longer samples**: More data, better estimation precision, but risk of including structural breaks.
- **Shorter samples**: More homogeneous, better reflects current regime, but fewer observations and less estimation precision.
- **Practical guidance**: For daily equity GARCH, 3-5 years is often a good balance. For monthly Markov switching, 20-30 years is needed to observe enough regime transitions, even though structural breaks are more likely over this horizon.

## 9. Sample Size Requirements

### General Minimums

| Model | Minimum Observations | Recommended |
|---|---|---|
| OLS regression (k predictors) | 10*k to 20*k | 30*k or more |
| ARIMA(p,d,q) | 50 + 10*(p+q) | 200+ |
| GARCH(1,1) | 500 | 1000+ |
| GJR-GARCH / EGARCH | 500 | 1500+ |
| Markov Switching (2 regimes) | 150 | 300+ (monthly) |
| Markov Switching (3 regimes) | 300 | 500+ (monthly) |
| VAR(p) with k variables | 50 + 10*k*p | 200+ |

### Why These Minimums

- **GARCH**: The ARCH and GARCH coefficients sum to near 1.0, making the likelihood surface flat. Many observations are needed to identify the separate contributions of alpha and beta.
- **Markov Switching**: The model must observe enough regime transitions to estimate transition probabilities. With only 2 bear-market episodes in the sample, the transition probability P(bear -> bull) is poorly identified.
- **Tail estimation**: To estimate the 1% quantile (VaR), you need at least 100 exceedances. For a 1% threshold, that requires 10,000 observations minimum. With 2,500 daily observations (10 years), you have only ~25 exceedances at the 1% level -- barely enough.

### Consequences of Insufficient Data

- **Wide confidence intervals**: Parameter estimates have large standard errors.
- **Convergence failure**: Optimization may not converge, especially for GARCH and regime-switching models.
- **Unstable results**: Small changes in the sample cause large changes in estimates.
- **False negatives**: Tests lack power to detect real effects.

## 10. Ignoring Transaction Costs and Market Impact

### Not a Statistical Pitfall Per Se, But Consequential

When evaluating trading strategies derived from statistical models:

- **Transaction costs**: A strategy that trades daily incurs substantial costs (bid-ask spread, commissions, slippage). A strategy with Sharpe ratio 1.0 before costs may have Sharpe 0.3 after costs.
- **Market impact**: Large orders move prices. A signal that works for $1M may not work for $1B.
- **Estimated transaction costs**: For liquid US large-cap equities, assume 5-20 bps per round trip. For less liquid assets, 50-200 bps.

### Rule of Thumb

If a backtest shows annual turnover of 500% and gross alpha of 3%, assume net alpha is approximately zero after realistic transaction costs.

## Summary: The Pitfall Detection Checklist

Before trusting any quantitative result, ask:

1. Are all variables stationary, or is cointegration established? (Spurious regression)
2. How many specifications were tested? Is there a multiple testing correction? (Data snooping)
3. Does the data include failed/delisted entities? (Survivorship bias)
4. For every input, was it available at the decision time? (Look-ahead bias)
5. Are parameters stable across sub-periods? (Structural breaks)
6. Is the sample large enough for the model complexity? (Sample size)
7. Are predictors highly correlated? (Multicollinearity)
8. Are standard errors corrected for heteroscedasticity and autocorrelation? (Heteroscedasticity)
9. Are results robust out of sample? (Overfitting)
10. Are realistic transaction costs and market impact included? (Implementation gap)

If any of these checks fail, the results should be treated with appropriate skepticism and the limitation should be disclosed prominently.
