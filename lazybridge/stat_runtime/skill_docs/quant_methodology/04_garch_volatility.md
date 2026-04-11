# GARCH Volatility Modeling

## What GARCH Models
GARCH (Generalized Autoregressive Conditional Heteroscedasticity) models time-varying volatility in financial returns. They capture volatility clustering — the empirical fact that large returns tend to be followed by large returns.

## When to Use GARCH
- Returns show volatility clustering (periods of high/low volatility)
- You need conditional volatility forecasts (e.g., for VaR)
- Residuals from a mean model show ARCH effects (squared residuals are autocorrelated)
- Risk management, option pricing, portfolio optimization

## When NOT to Use GARCH
- Data is not a return series (use on stationary returns, not prices)
- Volatility is constant (check with Ljung-Box on squared returns)
- Sample size < 500 (GARCH needs sufficient data for reliable estimation)
- Structural breaks in volatility regime (consider Markov switching instead)

## GARCH(p,q) Specification
The conditional variance equation:
σ²_t = ω + α₁ε²_{t-1} + ... + αqε²_{t-q} + β₁σ²_{t-1} + ... + βpσ²_{t-p}

Where:
- ω (omega): long-run variance weight (must be > 0)
- α (alpha): ARCH terms — reaction to past shocks
- β (beta): GARCH terms — persistence of past volatility
- α + β < 1 required for stationarity (sum near 0.95-0.99 is typical)

### Common Specifications
| Model | Params | Use Case |
|-------|--------|----------|
| GARCH(1,1) | p=1, q=1 | Default starting point, works for most return series |
| GARCH(1,2) | p=1, q=2 | When shock effects persist for multiple periods |
| GARCH(2,1) | p=2, q=1 | When volatility persistence has two components |

### GARCH(1,1) is usually sufficient
In practice, GARCH(1,1) captures most of the conditional variance dynamics. Higher-order models rarely improve fit significantly. Start with (1,1) and only increase if diagnostics indicate poor fit.

## Interpretation Guide

### Parameters
- **High α (>0.15)**: Volatility reacts strongly to recent shocks
- **High β (>0.90)**: Volatility is highly persistent
- **α + β close to 1**: Long memory in volatility (typical for equities)
- **α + β > 1**: Explosive process — model misspecified

### Conditional Volatility Plot
- Spikes correspond to crisis periods / market stress
- Baseline level = unconditional volatility: ω / (1 - α - β)
- Compare against realized volatility for model adequacy

### Diagnostics Checklist
1. Ljung-Box on standardized residuals → no serial correlation
2. Ljung-Box on squared standardized residuals → no remaining ARCH effects
3. Jarque-Bera on standardized residuals → assess normality
4. If normality fails: re-estimate with Student-t distribution (dist="t")

## Value at Risk (VaR)
GARCH provides daily VaR:
- VaR_α = μ_t - z_α × σ_t
- Where σ_t is the conditional standard deviation
- z_α is the normal quantile (1.645 for 95%, 2.326 for 99%)

## Volatility Forecasting
- Short-horizon (1-5 days): GARCH forecasts are reliable
- Medium-horizon (5-20 days): accuracy degrades, forecasts revert to unconditional
- Long-horizon (>20 days): use unconditional volatility instead

## Common Pitfalls
1. **Fitting on price levels**: GARCH is for returns, not prices
2. **Ignoring stationarity**: Returns must be stationary
3. **Overfitting with higher orders**: GARCH(1,1) is almost always sufficient
4. **Ignoring leverage effect**: Negative returns often increase volatility more than positive (use EGARCH/GJR-GARCH)
5. **Normal distribution assumption**: Financial returns are fat-tailed — use t-distribution
