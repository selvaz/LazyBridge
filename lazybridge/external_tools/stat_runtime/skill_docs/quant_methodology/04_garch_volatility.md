# GARCH Volatility Modeling

## What GARCH Models Do

GARCH (Generalized Autoregressive Conditional Heteroscedasticity) models capture **time-varying volatility** in financial return series. The core insight: large shocks to returns tend to be followed by large shocks (of either sign), and small shocks tend to be followed by small shocks. This phenomenon is called **volatility clustering** and is one of the most robust empirical regularities in finance.

GARCH models estimate **conditional variance** -- the variance of returns at time t, given all information available up to time t-1. This is distinct from unconditional (sample) variance, which is a single number for the entire series.

## When to Use GARCH

Use GARCH modeling when:

- You need to estimate time-varying risk (conditional volatility) for portfolio management or risk budgeting.
- You need Value-at-Risk (VaR) or Expected Shortfall (ES) estimates that adapt to current market conditions.
- You observe volatility clustering in return series (check the ACF of squared returns -- significant autocorrelation indicates GARCH effects).
- You want to produce volatility forecasts for option pricing, hedging ratios, or position sizing.
- Residuals from a mean model (e.g., ARMA) show heteroscedasticity (Engle's ARCH-LM test rejects).

Do NOT use GARCH when:

- Your data is not a return series or a stationary series. GARCH is not for levels of prices, GDP, or other integrated processes.
- You have fewer than ~500 observations. GARCH estimation requires substantial data; with daily data, aim for at least 2 years (500+ obs), preferably 5+ years.
- You have ultra-high-frequency (tick) data. Use realized volatility estimators instead (see section below).
- The squared returns show no significant autocorrelation. If there is no ARCH effect, a constant-variance model suffices.

## The GARCH(p,q) Specification

### Mean Equation

Most GARCH models include a mean equation for returns:

```
r_t = mu + epsilon_t
epsilon_t = sigma_t * z_t,   z_t ~ D(0,1)
```

where `D` is typically Normal, Student-t, or Skewed Student-t. The choice of distribution matters for tail risk estimation.

### Variance Equation: GARCH(1,1)

The workhorse model is GARCH(1,1):

```
sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
```

- **omega** (> 0): the baseline variance level. Determines the unconditional variance.
- **alpha** (>= 0): the ARCH coefficient. Measures how much yesterday's shock affects today's variance. Typical range: 0.03-0.15 for daily equity returns.
- **beta** (>= 0): the GARCH coefficient. Measures persistence of variance. Typical range: 0.80-0.95 for daily equity returns.
- **alpha + beta**: the persistence parameter. Must be < 1 for covariance stationarity. Values above 0.98 suggest extremely persistent volatility (close to an IGARCH process).

### Interpreting Parameters

| Parameter | Typical (daily equity) | Meaning |
|---|---|---|
| omega | 0.000001 - 0.00001 | Long-run variance floor |
| alpha | 0.03 - 0.15 | Shock sensitivity |
| beta | 0.80 - 0.95 | Volatility memory |
| alpha + beta | 0.95 - 0.99 | Total persistence |

**Unconditional variance**: `sigma^2 = omega / (1 - alpha - beta)`. If alpha=0.08, beta=0.90, omega=0.000002, then unconditional variance = 0.000002 / 0.02 = 0.0001, implying unconditional annualized volatility of sqrt(0.0001 * 252) ~ 15.9%.

**Half-life of volatility shocks**: `log(0.5) / log(alpha + beta)`. For persistence = 0.97, half-life ~ 23 days. For persistence = 0.99, half-life ~ 69 days.

### Higher-Order GARCH(p,q)

GARCH(2,1) or GARCH(1,2) are occasionally useful but rarely improve on GARCH(1,1) for daily financial returns. The literature strongly suggests GARCH(1,1) is adequate for the vast majority of financial applications. If GARCH(1,1) fits poorly, try a different variance specification (EGARCH, GJR) before increasing p or q.

## Asymmetric GARCH Models

### The Leverage Effect

Equity returns exhibit an asymmetry: negative returns increase future volatility more than positive returns of the same magnitude. This is called the **leverage effect** (though the mechanism is debated -- it may be driven by volatility feedback rather than actual leverage changes).

Standard GARCH(1,1) cannot capture this asymmetry because it uses `epsilon^2`, which is symmetric. Two extensions address this:

### GJR-GARCH (Glosten-Jagannathan-Runkle)

```
sigma^2_t = omega + (alpha + gamma * I_{t-1}) * epsilon^2_{t-1} + beta * sigma^2_{t-1}
```

where `I_{t-1} = 1` if `epsilon_{t-1} < 0` (bad news), `0` otherwise.

- **gamma > 0** confirms a leverage effect. Typical values for equity indices: 0.05-0.15.
- The impact of bad news is `alpha + gamma`; good news impact is `alpha`.
- News impact ratio: `(alpha + gamma) / alpha`. For equity indices, this ratio is typically 2-4x, meaning negative shocks have 2-4 times the volatility impact of positive shocks.

### EGARCH (Exponential GARCH)

```
log(sigma^2_t) = omega + alpha * |z_{t-1}| + gamma * z_{t-1} + beta * log(sigma^2_{t-1})
```

Advantages over GJR-GARCH:
- Models log-variance, so conditional variance is always positive (no parameter constraints needed).
- Can capture asymmetry through the `gamma` parameter (gamma < 0 indicates leverage effect).
- More flexible functional form.

Disadvantage:
- Log-variance parameterization makes unconditional variance harder to compute analytically.
- Slightly more difficult to interpret parameters directly.

### Which Asymmetric Model to Choose

- For equity returns: **always try an asymmetric model**. The leverage effect is well-established for equities and equity indices.
- For FX returns: leverage effects are weaker and often insignificant. Standard GARCH(1,1) may suffice.
- For commodity returns: leverage effects vary by commodity. Energy tends to show asymmetry; metals less so.
- If gamma is statistically insignificant (p > 0.05), revert to symmetric GARCH(1,1).

## Error Distributions

The choice of innovation distribution `D(0,1)` matters greatly for tail risk:

| Distribution | When to Use | Typical df |
|---|---|---|
| Normal | Rarely appropriate for financial returns | n/a |
| Student-t | Default choice; captures fat tails | 4-8 for equities |
| Skewed Student-t | When returns are asymmetrically distributed | df=4-8, skew=-0.1 to -0.3 |
| GED | Alternative to Student-t | 1.0-1.8 |

**Key rule**: If the degrees-of-freedom parameter is estimated below 6, the Normal distribution is clearly inappropriate. For most equity return series, Student-t with df between 4 and 8 is standard.

Always check: after fitting, do the standardized residuals `z_t = epsilon_t / sigma_t` look like the assumed distribution? Use a QQ-plot and Kolmogorov-Smirnov test.

## Diagnostics After Fitting

### Mandatory Checks

1. **Ljung-Box test on standardized residuals** (`z_t`): Should show NO significant autocorrelation. If it does, the mean model is misspecified.
2. **Ljung-Box test on squared standardized residuals** (`z_t^2`): Should show NO significant autocorrelation. If it does, the variance model is misspecified (consider higher-order GARCH or asymmetric specification).
3. **ARCH-LM test on standardized residuals**: Should NOT reject. If it rejects, remaining ARCH effects exist.
4. **Jarque-Bera test on standardized residuals**: If using Normal distribution and this rejects, switch to Student-t.
5. **Sign bias test**: Tests for asymmetric effects. If significant, switch to GJR-GARCH or EGARCH.
6. **Parameter significance**: All parameters should be statistically significant (p < 0.05). Insignificant parameters suggest over-parameterization.

### What to Do When Diagnostics Fail

| Diagnostic Failure | Action |
|---|---|
| Autocorrelation in z_t | Add AR/MA terms to the mean equation |
| Autocorrelation in z_t^2 | Try higher-order GARCH or asymmetric model |
| ARCH-LM rejects | Variance model inadequate; try EGARCH |
| Sign bias significant | Switch to GJR-GARCH or EGARCH |
| QQ-plot shows fat tails | Switch from Normal to Student-t distribution |
| alpha + beta >= 1.0 | Data may have structural breaks; consider sub-period estimation |

## Value-at-Risk Calculation

### Parametric VaR from GARCH

Given the one-step-ahead conditional volatility forecast `sigma_{t+1}`:

```
VaR_{alpha} = -mu - sigma_{t+1} * q_{alpha}(D)
```

where `q_{alpha}(D)` is the alpha-quantile of the innovation distribution.

For a 1% VaR with Normal distribution: `q_{0.01} = -2.326`
For a 1% VaR with Student-t(5): `q_{0.01} = -3.365`
For a 1% VaR with Student-t(8): `q_{0.01} = -2.896`

**The distribution choice dramatically affects VaR.** A GARCH-Normal model at 1% VaR will underestimate risk by 20-40% compared to GARCH-t(5) when the true distribution is fat-tailed.

### Multi-Step VaR

For h-step-ahead VaR, either:
1. Use the GARCH model to recursively forecast `sigma^2_{t+h}` (analytical formulas exist for GARCH(1,1)).
2. Use simulation: draw many paths of `z_t` from the fitted distribution, propagate through the GARCH equations, and take the empirical quantile.

Simulation is preferred for h > 5 because analytical multi-step formulas assume away mean reversion effects that matter at longer horizons.

### VaR Backtesting

Always backtest VaR estimates:
- **Kupiec test** (unconditional coverage): Does the realized violation rate match the target? At 1% VaR, expect ~1% violations.
- **Christoffersen test** (conditional coverage): Are violations independent? Clustered violations indicate the model fails to adapt quickly enough.
- A good 1% VaR model should produce 0.5%-1.5% violations with no clustering.

## GARCH vs Realized Volatility

| Feature | GARCH | Realized Volatility |
|---|---|---|
| Data requirement | Daily returns only | High-frequency (intraday) data |
| Estimation | Parametric (MLE) | Non-parametric (sum of squared intraday returns) |
| Microstructure noise | Not affected | Requires noise correction (e.g., kernel-based) |
| Forecasting | Built-in via model dynamics | Requires separate forecast model (HAR-RV) |
| Model risk | Sensitive to specification | Less model-dependent |
| Availability | Any daily return series | Only if intraday data exists |

**Decision rule**: If you have reliable 5-minute or 1-minute return data, realized volatility (with the HAR-RV model for forecasting) generally outperforms GARCH for volatility forecasting. If you only have daily data, GARCH is the standard approach.

## Common Pitfalls

### 1. Fitting GARCH to Non-Stationary Data

GARCH requires the return series to be stationary. If you accidentally fit GARCH to price levels or a non-stationary series, you will get meaningless persistence parameters (alpha + beta ~ 1.0) and the model will appear to fit but will not forecast well. Always transform prices to log-returns first.

### 2. Ignoring the Mean Model

The mean equation matters. Setting `mu = 0` is often appropriate for daily returns, but for lower-frequency data or series with significant autocorrelation, include AR terms. A misspecified mean model contaminates the variance estimates.

### 3. Overfitting with Complex Specifications

GARCH(1,1) is almost always sufficient. Do not default to GARCH(2,2) or elaborate component models without diagnostic evidence that GARCH(1,1) is inadequate. Extra parameters consume degrees of freedom and can cause convergence issues.

### 4. Misinterpreting Conditional vs Unconditional Volatility

The conditional volatility `sigma_t` is the model's estimate of volatility AT TIME t. The unconditional volatility is the long-run average. Do not report conditional volatility as if it were a stable characteristic of the asset. It changes every period.

### 5. Ignoring the Leverage Effect for Equities

For equity returns, always test for asymmetry. Reporting a symmetric GARCH(1,1) for equity index returns without checking for leverage effects is a methodological oversight.

### 6. Wrong Annualization

Daily conditional variance `sigma^2_t` is annualized as `sigma^2_t * 252` (variance scales linearly with time under iid assumptions). Conditional volatility (standard deviation) annualizes as `sigma_t * sqrt(252)`. Do not annualize variance by multiplying by `sqrt(252)`.

### 7. Convergence Failures

GARCH estimation uses numerical optimization (typically BFGS or SLSQP). Convergence failures often indicate:
- Outliers in the return series (consider winsorizing at 5 standard deviations).
- Structural breaks (the parameters are not constant over the sample).
- Poor starting values (try different initial parameter guesses).
- A fundamentally wrong model specification.

If the optimizer does not converge, do NOT use the parameter estimates.

## Practical Workflow Summary

1. Compute log-returns from prices.
2. Run ADF/KPSS to confirm stationarity.
3. Check ACF of squared returns for ARCH effects (or run Engle's ARCH-LM test).
4. Fit GARCH(1,1) with Student-t distribution as a baseline.
5. Check for leverage effects (sign bias test or fit GJR-GARCH and check gamma significance).
6. Run Ljung-Box on standardized and squared standardized residuals.
7. Compare AIC/BIC across candidate models.
8. Use the selected model for volatility forecasting or VaR calculation.
9. Backtest any VaR estimates out of sample.

## Parameter Reporting Conventions

When reporting GARCH results, always include:

- Model specification (e.g., GJR-GARCH(1,1) with Student-t innovations)
- All parameter estimates with standard errors and p-values
- alpha + beta persistence measure
- Unconditional annualized volatility implied by the model
- Log-likelihood, AIC, BIC
- Ljung-Box p-values for standardized and squared standardized residuals (at lag 10 and 20)
- Sample period and number of observations
- Any data transformations applied
