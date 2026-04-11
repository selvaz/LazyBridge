# Model Selection Guide

## Information Criteria

### AIC (Akaike Information Criterion)
- AIC = 2k - 2ln(L), where k = number of parameters, L = likelihood
- Balances fit quality against model complexity
- Lower AIC = better model
- Tends to favor slightly more complex models

### BIC (Bayesian Information Criterion)
- BIC = k·ln(n) - 2ln(L), where n = sample size
- Stronger penalty for complexity than AIC
- Lower BIC = better model
- Preferred when you want parsimony

### When They Disagree
- AIC picks the more complex model → it optimizes prediction
- BIC picks the simpler model → it identifies the true model
- For forecasting: prefer AIC
- For identifying the data-generating process: prefer BIC
- In practice: if difference < 2, models are essentially equivalent

## Decision Framework

### Step 1: Define the Question
- Forecasting? → optimize predictive accuracy (AIC, out-of-sample tests)
- Understanding dynamics? → identify true structure (BIC)
- Risk measurement? → optimize tail behavior (VaR backtests)

### Step 2: Estimate Candidates
For a return series, typical candidates:
1. OLS with trend/seasonal (baseline)
2. ARIMA(p,d,q) with varying orders
3. GARCH(1,1) if volatility clustering present
4. Markov switching if regime changes suspected

### Step 3: Compare Using Criteria
- Within the same family: use AIC/BIC
- Across families: use AIC/BIC + domain knowledge
- Always check that the best model passes diagnostics

### Step 4: Validate
- Reserve 20-30% of data for out-of-sample testing
- Compute forecast errors (MAE, RMSE)
- Check if in-sample winner is also out-of-sample winner

## Rules of Thumb

### AIC/BIC Differences
| Difference | Evidence |
|-----------|----------|
| 0-2 | Weak — models are essentially equivalent |
| 2-6 | Positive — some evidence for the better model |
| 6-10 | Strong — clear evidence |
| > 10 | Very strong — decisive |

### Parsimony Principle
- Simpler models generalize better
- If two models have similar AIC/BIC, prefer the simpler one
- Extra parameters that don't reduce AIC by >2 are not justified

### Overfitting Signals
- In-sample R² is high but out-of-sample R² is low
- AIC improves but BIC doesn't (or worsens)
- Parameters are statistically insignificant
- Residual diagnostics pass but forecast errors are large

## Model Family Selection Guide

| Observation | Suggested Model |
|-------------|----------------|
| Linear relationship, constant variance | OLS |
| Serial correlation in data | ARIMA |
| Volatility clustering in returns | GARCH |
| Regime-dependent behavior | Markov Switching |
| Serial correlation + volatility clustering | ARIMA-GARCH |
| Time-varying volatility + regime changes | Markov + GARCH diagnostic comparison |

## Common Mistakes
1. Comparing AIC across different samples (must use same data)
2. Using R² for time series model comparison (use AIC/BIC)
3. Selecting models based only on in-sample fit
4. Adding variables that are significant but don't improve information criteria
5. Ignoring diagnostic failures in the "best" model
