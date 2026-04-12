# Model Selection and Comparison

## The Core Problem

Model selection in econometrics and quantitative finance involves choosing among competing specifications that trade off **fit** (explaining observed data) against **complexity** (number of parameters). A model with more parameters will always fit the in-sample data at least as well, but may generalize poorly to new data. The goal is to find the model that best captures the true data-generating process, not the one that memorizes the training sample.

## Information Criteria

### AIC (Akaike Information Criterion)

```
AIC = -2 * log_likelihood + 2 * k
```

where `k` is the number of estimated parameters.

- AIC estimates the expected Kullback-Leibler divergence between the fitted model and the true model.
- AIC is asymptotically equivalent to leave-one-out cross-validation.
- **Lower AIC is better.**
- AIC tends to select larger models. It does not penalize complexity as heavily as BIC.
- AIC is consistent for prediction: it selects the model that minimizes expected out-of-sample prediction error.
- AIC is NOT consistent for model selection: even as sample size grows to infinity, AIC may select an overly complex model if the true model is among the candidates.

### BIC (Bayesian Information Criterion)

```
BIC = -2 * log_likelihood + k * log(n)
```

where `n` is the number of observations.

- BIC approximates the log marginal likelihood (Bayesian model evidence).
- The penalty `k * log(n)` grows with sample size, making BIC increasingly parsimonious for larger datasets.
- **Lower BIC is better.**
- For n >= 8, BIC penalizes complexity more heavily than AIC (since log(8) > 2).
- BIC is consistent for model selection: as n grows, BIC selects the true model (if it is among the candidates) with probability approaching 1.
- BIC is NOT optimal for prediction: it may select too simple a model for finite samples.

### AIC vs BIC: When to Use Which

| Criterion | Use When | Bias |
|---|---|---|
| AIC | Goal is best prediction on new data | Tends to overfit (select too many parameters) |
| BIC | Goal is identifying the true model | Tends to underfit (select too few parameters) |
| Both | Always report both; note agreement or disagreement | Agreement strengthens confidence |

**Practical guidance for quant finance**:

- If AIC and BIC agree on the best model, confidence is high.
- If they disagree, BIC's choice is typically the safer default for inference and reporting. AIC's choice may forecast better in the short run.
- For GARCH model selection (choosing between GARCH(1,1), GJR-GARCH(1,1), EGARCH(1,1)): BIC usually selects the most parsimonious adequate model.
- For ARIMA order selection: AIC tends to select higher orders than BIC. When in doubt, use BIC for the variance model and AIC for the mean model.
- The difference in AIC/BIC between models matters: a difference of less than 2 is weak evidence; 2-6 is positive evidence; 6-10 is strong evidence; above 10 is very strong evidence (Burnham & Anderson scale for AIC; Kass & Raftery scale for BIC).

### AICc (Corrected AIC)

```
AICc = AIC + (2k^2 + 2k) / (n - k - 1)
```

AICc applies a finite-sample correction. Use AICc instead of AIC when `n/k < 40`. For most financial applications with daily data (n > 1000), AIC and AICc are virtually identical. For small samples (quarterly macro data, 80-200 obs), AICc can make a meaningful difference.

### HQIC (Hannan-Quinn Information Criterion)

```
HQIC = -2 * log_likelihood + 2 * k * log(log(n))
```

HQIC falls between AIC and BIC in terms of penalty strength. It is consistent for model selection (like BIC) but penalizes less aggressively. Rarely used in practice; included here for completeness.

## Interpreting Information Criteria Values

### Absolute Values Are Meaningless

AIC = -3456.78 means nothing by itself. Only **differences** between models fitted to the **same data** are meaningful.

### Comparing Models

Always compare models fitted to:
1. The **same dataset** (same observations, same time period).
2. The **same dependent variable** (do not compare AIC of a model for returns vs a model for log-returns).
3. The **same sample size** (if models use different lag lengths, they may effectively have different sample sizes due to initial observations lost to lags -- adjust for this).

### Delta-AIC Table

Compute `delta_i = AIC_i - AIC_min` for each model. Then:

| delta_i | Interpretation |
|---|---|
| 0-2 | Substantial support; model is competitive |
| 2-4 | Some support; model is plausible |
| 4-7 | Less support; model is unlikely best |
| 7-10 | Very little support |
| > 10 | Essentially no support; discard this model |

### Akaike Weights

For more nuanced comparison, compute Akaike weights:

```
w_i = exp(-0.5 * delta_i) / sum_j exp(-0.5 * delta_j)
```

These sum to 1 and can be interpreted as approximate posterior model probabilities. A model with w_i = 0.73 has roughly 73% probability of being the best model (in the K-L sense) among the candidates.

## In-Sample vs Out-of-Sample Validation

### The Fundamental Tension

In-sample metrics (R-squared, log-likelihood, information criteria) evaluate how well the model explains the data it was trained on. Out-of-sample metrics evaluate how well the model predicts data it has never seen. For financial applications, out-of-sample performance is almost always what matters.

### Why In-Sample Metrics Mislead

- **Overfitting**: A model with enough parameters can fit noise as well as signal. In-sample R-squared always increases (or stays the same) as you add parameters.
- **Data snooping**: If you tried 20 specifications and picked the best in-sample, you are implicitly overfitting to the specific sample.
- **Structural change**: Financial data exhibits regime shifts. A model that fits 2005-2015 perfectly may fail in 2016-2020 due to changed market dynamics.

### Out-of-Sample Testing for Time Series

**Expanding window** (preferred for most financial applications):
1. Fit the model on data from time 1 to time T.
2. Forecast time T+1.
3. Expand the window: fit on data from time 1 to time T+1.
4. Forecast time T+2.
5. Continue until the end of the sample.

**Rolling window** (preferred when you suspect structural change):
1. Fit the model on data from time t to time t+W (window of size W).
2. Forecast time t+W+1.
3. Roll: fit on data from time t+1 to time t+W+1.
4. Forecast time t+W+2.
5. Continue until the end of the sample.

**Key design choices**:
- Reserve at least 20-30% of the sample for out-of-sample evaluation.
- For daily data with 2500 observations (10 years): use first 1500 for initial training, last 1000 for evaluation.
- For monthly data with 360 observations (30 years): use first 240 for training, last 120 for evaluation.
- Rolling window size: typically 5-10 years for daily data; 15-20 years for monthly.

### Out-of-Sample Metrics

| Metric | Formula | Use Case |
|---|---|---|
| RMSE | sqrt(mean(e_t^2)) | General forecast accuracy |
| MAE | mean(abs(e_t)) | Robust to outliers |
| MAPE | mean(abs(e_t / y_t)) * 100 | Scale-free comparison |
| Directional accuracy | mean(sign(forecast_t) == sign(actual_t)) | Sign prediction |
| Out-of-sample R^2 | 1 - sum(e_t^2) / sum((y_t - y_bar)^2) | Improvement over historical mean |
| Log-likelihood (OOS) | sum(log f(y_t | model)) | Density forecast quality |

**Critical note on out-of-sample R-squared**: For return forecasting, out-of-sample R^2 values of 0.5%-3.0% are considered economically meaningful (Campbell & Thompson, 2008). Do not expect high R-squared for return prediction. For volatility forecasting, R-squared of 20%-50% against realized volatility is typical for good GARCH models.

## Cross-Validation for Time Series

### Why Standard K-Fold CV Fails

Standard k-fold cross-validation randomly splits data into folds, destroying the temporal ordering. This creates look-ahead bias: the model trains on future data to predict past data. For time series, **NEVER use random cross-validation.**

### Time Series Cross-Validation (Walk-Forward)

This is the expanding or rolling window approach described above. It preserves temporal ordering and simulates real forecasting conditions.

### Blocked Time Series CV

A compromise that provides more test sets:
1. Split the time series into K consecutive, non-overlapping blocks.
2. For each block k=2,...,K: train on blocks 1 to k-1, test on block k.
3. Average the test metrics across all folds.

Include a **gap** between training and test blocks (e.g., skip 5 observations) to avoid information leakage from autocorrelation.

### Purged Cross-Validation

For applications where data points have overlapping information (e.g., overlapping return windows), use purged CV (de Prado, 2018):
1. Remove from the training set any observations whose information set overlaps with the test set.
2. Add an embargo period after each test set to prevent leakage.

This is especially important for features computed from rolling windows (e.g., 20-day moving average).

## The Parsimony Principle

### Occam's Razor in Econometrics

When two models provide similar fit, prefer the simpler one. This is not merely aesthetic -- it reflects:

1. **Estimation precision**: Fewer parameters means each parameter is estimated with more data, reducing standard errors.
2. **Robustness to structural change**: Simple models degrade gracefully; complex models can fail catastrophically when the data-generating process shifts.
3. **Interpretability**: A GARCH(1,1) with 3 variance parameters is interpretable. A GARCH(3,3) with 7 variance parameters is not.
4. **Forecasting stability**: Complex models produce forecasts with higher variance. The bias-variance tradeoff favors simpler models for out-of-sample prediction.

### Practical Parsimony Rules

- ARIMA: Prefer total order (p+d+q) <= 5. If you need ARIMA(4,1,3), something is likely wrong.
- GARCH: Almost never go beyond GARCH(1,1). Try asymmetric specifications before higher orders.
- Regression: If adding a variable improves R-squared by less than 1 percentage point and is not theoretically motivated, leave it out.
- Regimes: 2 regimes is the default. Use 3 only with strong evidence and economic justification.

## Overfitting Detection

### Symptoms of Overfitting

1. **Large gap between in-sample and out-of-sample performance**: If in-sample R^2 = 0.15 but out-of-sample R^2 = -0.02, the model is overfitting.
2. **Unstable parameter estimates**: If re-estimating on a slightly different sample (e.g., dropping the last 6 months) changes coefficients substantially, the model is fragile.
3. **Suspiciously good in-sample fit for noisy data**: Daily return R^2 above 5% from a linear model should be viewed with extreme skepticism.
4. **Many insignificant parameters**: If half the coefficients have p > 0.10, the model is over-parameterized.
5. **Information criteria disagree**: If AIC selects a larger model but BIC selects a smaller one, the larger model is likely fitting noise.

### How to Diagnose

1. **Compare training vs holdout error**: Split the sample. If training error is much lower than holdout error, the model overfits.
2. **Stability analysis**: Estimate the model on sub-periods. If parameter estimates differ wildly, the model is capturing sample-specific patterns.
3. **Regularization comparison**: If adding an L1 or L2 penalty shrinks many coefficients toward zero, those coefficients were fitting noise.
4. **Progressive complexity test**: Start with the simplest model. Add complexity one step at a time. Stop when out-of-sample performance no longer improves.

## Nested Model Comparison

### Likelihood Ratio Test

For nested models (Model A is a special case of Model B):

```
LR = -2 * (log_likelihood_A - log_likelihood_B)
```

Under the null hypothesis that Model A is correct, `LR ~ chi^2(k_B - k_A)`, where the difference is in the number of parameters.

**Example**: Testing GARCH(1,1) vs GJR-GARCH(1,1) (which adds one parameter, gamma):
- LR = -2 * (LL_garch - LL_gjr)
- Under H0, LR ~ chi^2(1)
- Reject at 5% if LR > 3.84

**Caution**: The LR test requires the restricted model to be nested within the unrestricted model. You cannot use it to compare GARCH(1,1) with EGARCH(1,1) because neither is nested within the other. Use information criteria instead.

### When LR Tests Are Invalid

1. **Regime-switching models**: Testing k regimes vs k+1 regimes violates regularity conditions (nuisance parameters on the boundary). Use BIC or simulation-based tests instead.
2. **Unit root boundary**: Testing ARMA vs ARIMA when the AR root is near unity requires non-standard critical values.
3. **Non-nested models**: GARCH vs EGARCH, or Normal vs Student-t distribution. Use Vuong test or information criteria.

### Vuong Test for Non-Nested Models

For non-nested models, the Vuong (1989) test compares the average log-likelihood ratio:

```
LR_i = log f_A(y_i) - log f_B(y_i)   (pointwise log-likelihood difference)
V = sqrt(n) * mean(LR_i) / std(LR_i)
```

Under the null of equivalent fit, `V ~ N(0,1)`. Reject in favor of Model A if V > 1.96; reject in favor of Model B if V < -1.96.

## Multi-Model Comparison Workflow

### Step 1: Define the Candidate Set

Choose models with clear theoretical motivation. Do NOT include every possible specification. A reasonable candidate set for equity return volatility:

1. GARCH(1,1) with Normal innovations
2. GARCH(1,1) with Student-t innovations
3. GJR-GARCH(1,1) with Student-t innovations
4. EGARCH(1,1) with Student-t innovations

### Step 2: Estimate All Models on the Same Data

Ensure identical sample periods and identical dependent variables.

### Step 3: In-Sample Comparison

Report a table:

| Model | Log-Lik | k | AIC | BIC | alpha+beta | df |
|---|---|---|---|---|---|---|
| GARCH-N | -3456.7 | 3 | 6919.4 | 6936.2 | 0.97 | - |
| GARCH-t | -3421.3 | 4 | 6850.6 | 6873.0 | 0.96 | 6.2 |
| GJR-t | -3415.8 | 5 | 6841.6 | 6869.6 | 0.96 | 6.5 |
| EGARCH-t | -3416.1 | 5 | 6842.2 | 6870.2 | - | 6.4 |

### Step 4: Out-of-Sample Comparison

Run expanding-window forecast evaluation. Report RMSE, MAE, and directional accuracy for each model.

### Step 5: Select and Justify

Choose the model with the best combination of:
- Lowest BIC (or AIC if prediction is the goal)
- Best out-of-sample forecast performance
- Adequate diagnostics (no remaining ARCH effects, no autocorrelation in residuals)
- Interpretable and stable parameters

If in-sample and out-of-sample rankings conflict, trust the out-of-sample ranking.

## Summary of Decision Rules

| Situation | Decision Rule |
|---|---|
| AIC and BIC agree | High confidence in that model |
| AIC and BIC disagree | Default to BIC for inference; check OOS performance |
| Delta-AIC < 2 between top models | Models are essentially equivalent; prefer simpler |
| OOS R^2 is negative | Model is worse than the historical mean; do not use |
| Adding parameter improves AIC but not BIC | Parameter is likely fitting noise; exclude unless theoretically motivated |
| Nested model test: p < 0.01 | Strong evidence for the more complex model |
| Nested model test: p between 0.01-0.05 | Evidence for complex model, but check OOS robustness |
| Nested model test: p > 0.10 | Insufficient evidence; keep the simpler model |
