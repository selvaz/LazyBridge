# Risk Analysis

## Value at Risk (VaR)

### Definition
VaR answers: "What is the maximum loss I can expect over a given horizon at a given confidence level?"

From a GARCH model:
```
VaR_α(t) = -(μ_t - z_α × σ_t)
```
where σ_t is the GARCH conditional standard deviation and z_α is the quantile of the assumed distribution.

### Common Confidence Levels

| Level | z (Normal) | z (Student-t, df=5) | Use Case |
|---|---|---|---|
| 95% | 1.645 | 2.015 | Internal risk monitoring |
| 99% | 2.326 | 3.365 | Regulatory (Basel III) |
| 99.5% | 2.576 | 4.032 | Stress testing |

### Interpretation Example
"The 1-day 99% VaR is 2.3%" means: on 99% of days, the portfolio will not lose more than 2.3%. Equivalently, we expect losses exceeding 2.3% about 2-3 days per year (1% × 252 trading days).

### VaR Does Not Tell You
- **How bad it can get beyond VaR**: VaR says nothing about the size of losses in the remaining (1-α) tail.
- **The distribution of tail losses**: A 99% VaR of 2.3% could have a worst case of 3% or 15%.

## Expected Shortfall (CVaR)

### Definition
Expected Shortfall answers: "When losses exceed VaR, what is the average loss?"

```
ES_α = -E[r | r < -VaR_α]
```

### Why ES is Preferred Over VaR
1. **Subadditivity**: ES satisfies all properties of a coherent risk measure. VaR does not — portfolio VaR can exceed the sum of individual VaRs.
2. **Tail information**: ES captures the severity of tail events, not just their threshold.
3. **Regulatory trend**: Basel III FRTB moved from VaR to ES for market risk capital.

### Typical ES-to-VaR Ratios
For equity returns with Student-t distribution (df=5):
- ES_99% ≈ 1.4 × VaR_99%
- ES_95% ≈ 1.5 × VaR_95%

For Normal distribution:
- ES_99% ≈ 1.14 × VaR_99%
- ES_95% ≈ 1.25 × VaR_95%

Fat-tailed distributions produce larger ES-to-VaR ratios because tail events are more severe.

## VaR from GARCH Models

### Daily VaR Calculation
Given a fitted GARCH model with conditional mean μ_t and conditional volatility σ_t:

```
Dollar_VaR = Portfolio_Value × (μ_t - z_α × σ_t)
```

### Multi-Day VaR (Square-Root-of-Time Rule)
```
VaR_α(h days) ≈ VaR_α(1 day) × sqrt(h)
```

This is exact for IID returns and approximate for GARCH. For high-persistence GARCH, compute the h-step-ahead variance forecast directly rather than using the square-root rule.

### Regulatory 10-Day VaR (Basel)
```
VaR_10day ≈ VaR_1day × sqrt(10) ≈ VaR_1day × 3.16
```

## VaR Backtesting

### Purpose
Check whether the VaR model is correctly calibrated. At the 99% level, expect approximately 1% of observations to exceed VaR.

### Kupiec Test (Unconditional Coverage)
- Null hypothesis: violation rate = (1-α).
- If p < 0.05: model is miscalibrated.
- Too many violations → VaR too low (underestimates risk).
- Too few violations → VaR too high (overestimates risk, capital inefficient).

### Christoffersen Test (Independence)
- Tests whether violations are independent over time.
- Clustered violations mean the GARCH model is too slow to adapt to regime changes.

### Traffic Light System (Basel)

| Violations in 250 days | Zone | Action |
|---|---|---|
| 0-4 | Green | Model accepted |
| 5-9 | Yellow | Increased scrutiny |
| 10+ | Red | Model rejected, capital surcharge |

For 99% VaR over 250 trading days, expected violations = 2.5.

## Regime-Dependent Risk

### Why Regimes Matter
Unconditional VaR averages across all market states. Risk differs dramatically by regime:

| Metric | Bull Regime | Bear Regime |
|---|---|---|
| Daily vol (annualized) | 12-16% | 25-45% |
| 99% VaR (daily) | 1.5-2.0% | 3.5-6.0% |
| Expected duration | 20-40 months | 5-15 months |

### Using Markov Switching for Risk
1. Fit a 2-regime Markov model.
2. Compute smoothed regime probability at current time.
3. Compute regime-conditional VaR: VaR_bull and VaR_bear.
4. Regime-weighted VaR = P(bull) × VaR_bull + P(bear) × VaR_bear.

This produces VaR that adapts to the current regime, not just current volatility.

## Volatility Forecasting for Risk

### Horizon and Accuracy
- **1-day ahead**: GARCH reliable. R² vs realized vol: 30-50%.
- **5-day ahead**: Still useful. R² drops to 20-35%.
- **20-day ahead**: Reverts toward unconditional. R² drops to 10-20%.
- **>20 days**: Use unconditional volatility.

### GARCH vs Realized Volatility
- Daily risk monitoring → GARCH (model-based, forward-looking).
- Monthly risk budgeting → realized volatility (backward-looking, non-parametric).
- Best practice: GARCH forecast + realized vol as a cross-check.

## Portfolio Risk

### Individual vs Portfolio Volatility
```
σ_portfolio = sqrt(w' Σ w)
```
where w = weight vector, Σ = covariance matrix.

Diversification benefit: σ_portfolio < sum of weighted individual σ's (unless all correlations = 1).

### Correlation Instability
Correlations increase during crises. A diversified portfolio in normal markets may concentrate risk during stress.

**Practical implication**: Stress-test portfolio VaR using crisis-period correlations, not full-sample.

## Practical Risk Rules

1. **Always use fat-tailed distributions** for VaR. Normal VaR understates tail risk by 30-50% for equities.
2. **Backtest regularly**: A model that passed last year may fail this year.
3. **Report both VaR and ES**: VaR for threshold, ES for severity.
4. **Use conditional VaR**: GARCH-based VaR adapts to current conditions.
5. **Be skeptical of very low VaR**: If 99% VaR is 0.5% for a single stock, the model is wrong.
6. **Square-root-of-time is approximate**: For persistent volatility, compute the h-step forecast directly.
7. **Know your regime**: Bear-market VaR is 2-3x bull-market VaR.
