# Writing Disciplined Research Reports

## Purpose of This Guide

This document specifies how to structure and write a quantitative finance research report. The goal is disciplined communication: every claim must be grounded in evidence, every limitation must be acknowledged, and the reader should be able to assess exactly how much confidence to place in each finding.

## Report Structure

### 1. Executive Summary (5-10 sentences)

The executive summary is the single most important section. Many readers will read only this. It must contain:

- **What was analyzed**: Asset(s), time period, data frequency.
- **What method was used**: One sentence naming the model family (e.g., "GJR-GARCH(1,1) with Student-t innovations").
- **Key finding**: The primary quantitative result (e.g., "Conditional volatility averaged 18.3% annualized, with two distinct high-volatility episodes").
- **Practical implication**: What the finding means for the reader's decision (e.g., "Current volatility is 1.5 standard deviations above the long-run average, suggesting elevated risk").
- **Key caveat**: The single most important limitation (e.g., "The model assumes no structural breaks in volatility dynamics; a regime shift would invalidate the forecast").

Do NOT include:
- Technical details (parameter values, test statistics).
- Multiple competing findings that confuse the reader.
- Speculative interpretations not supported by the analysis.

### 2. Data Description

Specify completely:

| Element | Example | Why It Matters |
|---|---|---|
| Asset / series name | S&P 500 Total Return Index | Identifies what was analyzed |
| Source | Bloomberg, CRSP, Yahoo Finance | Reproducibility |
| Time period | 2005-01-03 to 2024-12-31 | Defines the sample |
| Frequency | Daily close-to-close | Affects model choice |
| Number of observations | 5,032 trading days | Sample size adequacy |
| Transformations applied | Log returns: r_t = ln(P_t/P_{t-1}) | Reproducibility |
| Missing data treatment | 3 missing values filled by linear interpolation | Transparency |
| Outlier treatment | None / Winsorized at +/- 5 sigma | Affects tail estimates |

Include summary statistics:

| Statistic | Value |
|---|---|
| Mean (annualized) | 9.8% |
| Std dev (annualized) | 18.2% |
| Skewness | -0.42 |
| Excess kurtosis | 7.31 |
| Min daily return | -11.98% (2020-03-16) |
| Max daily return | +9.38% (2020-03-24) |
| Jarque-Bera p-value | < 0.001 |

The skewness and kurtosis numbers are critical: they justify the choice of fat-tailed distributions and asymmetric models.

### 3. Methodology

Describe the model specification precisely enough that another analyst could replicate it. Include:

- **Model equation(s)**: Write out the full mathematical specification. For GARCH(1,1):
  ```
  r_t = mu + epsilon_t
  epsilon_t = sigma_t * z_t, z_t ~ t(nu)
  sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
  ```
- **Estimation method**: Maximum likelihood, quasi-maximum likelihood, etc.
- **Software/library**: "Estimated using the `arch` Python package, version 6.x."
- **Pre-estimation diagnostics**: ADF and KPSS test results confirming stationarity. ARCH-LM test confirming the presence of ARCH effects.
- **Model selection rationale**: Why this specification was chosen over alternatives. Reference the information criteria comparison if applicable.

Do NOT:
- Describe the general theory of GARCH at length. The reader knows what GARCH is. State the specific model and move on.
- Omit the innovation distribution. The distribution choice affects every downstream result.

### 4. Results

#### Parameter Estimates

Always present in a table with standard errors and significance:

| Parameter | Estimate | Std Error | t-stat | p-value |
|---|---|---|---|---|
| mu | 0.0004 | 0.0001 | 3.21 | 0.001 |
| omega | 1.83e-06 | 4.12e-07 | 4.44 | <0.001 |
| alpha | 0.074 | 0.012 | 6.17 | <0.001 |
| beta | 0.912 | 0.013 | 70.15 | <0.001 |
| nu (df) | 6.42 | 0.87 | 7.38 | <0.001 |

Report derived quantities:
- Persistence: alpha + beta = 0.986
- Half-life of shocks: log(0.5) / log(0.986) = 49.2 days
- Unconditional annualized volatility: sqrt(omega / (1 - alpha - beta) * 252) = 17.1%

#### Model Fit Statistics

| Metric | Value |
|---|---|
| Log-likelihood | -7,234.5 |
| AIC | 14,479.0 |
| BIC | 14,511.7 |
| Observations | 5,032 |

#### Key Findings (Narrative)

Translate the numbers into interpretable statements:

- "Volatility persistence is very high (0.986), meaning shocks to volatility decay slowly with a half-life of approximately 49 trading days."
- "The degrees-of-freedom parameter (6.42) confirms fat tails in the return distribution, consistent with the excess kurtosis of 7.31 observed in the raw data."
- "The model implies a long-run annualized volatility of 17.1%, close to the sample standard deviation of 18.2%."

#### Forecasts

If forecasts are produced, report them with confidence intervals:

| Horizon | Forecast (Ann. Vol) | 95% CI Lower | 95% CI Upper |
|---|---|---|---|
| 1 day | 14.2% | - | - |
| 5 days | 14.8% | 11.3% | 19.1% |
| 20 days | 15.6% | 10.8% | 21.4% |

Note: 1-step-ahead point forecasts do not have meaningful confidence intervals from standard GARCH. Multi-step intervals should be obtained by simulation.

### 5. Diagnostics

This section is non-negotiable. Every report must include:

#### Residual Tests

| Test | Statistic | p-value | Result |
|---|---|---|---|
| Ljung-Box(10) on z_t | 11.3 | 0.334 | Pass (no autocorrelation) |
| Ljung-Box(20) on z_t | 22.1 | 0.335 | Pass |
| Ljung-Box(10) on z_t^2 | 8.7 | 0.562 | Pass (no remaining ARCH) |
| Ljung-Box(20) on z_t^2 | 17.4 | 0.628 | Pass |
| ARCH-LM(10) | 9.1 | 0.521 | Pass |
| Sign Bias (joint) | 4.2 | 0.241 | Pass (no asymmetry) |

**Interpretation rule**: If any of these tests reject at the 5% level, the model has a specification problem. State this clearly and explain what it implies. Do not bury a failed diagnostic in a table and hope the reader does not notice.

#### Visual Diagnostics

Always include or reference:
1. **Standardized residual time series**: Check for remaining patterns or outliers.
2. **ACF of standardized residuals**: Should show no significant spikes.
3. **ACF of squared standardized residuals**: Should show no significant spikes.
4. **QQ-plot of standardized residuals** against the assumed distribution: Points should lie on the 45-degree line.
5. **Conditional volatility time series**: Should show plausible dynamics and capture known high-volatility episodes.

### 6. Caveats and Limitations

Every report must have this section. It is not optional. Include:

#### Model Assumptions

State the assumptions and assess their validity:

- "The model assumes covariance stationarity of the return process. The ADF test (p < 0.001) supports this for the sample period, but structural breaks could invalidate this assumption."
- "The model assumes constant transition probabilities [for Markov switching]. If the regime-switching mechanism itself has changed (e.g., due to policy changes), this assumption is violated."
- "Parameters are assumed constant over the entire sample (2005-2024). A 19-year sample spans multiple market regimes, and parameter instability is possible."

#### Sample Limitations

- "The sample includes one major financial crisis (2008-2009) and one pandemic shock (2020). Results are heavily influenced by these episodes."
- "With 5,032 observations, asymptotic inference is generally reliable, but tail quantile estimates (1% VaR) are based on relatively few extreme observations."

#### Forecast Limitations

- "Volatility forecasts beyond 20 days should be treated as indicative, not precise. The GARCH model's forecast converges to the unconditional volatility at longer horizons."
- "The model cannot predict regime changes (sudden volatility spikes from unforeseen events)."

## Statistical Significance Language

### What to Say

| p-value | Language |
|---|---|
| p < 0.001 | "Highly statistically significant" or "strong evidence against the null" |
| 0.001 <= p < 0.01 | "Statistically significant at the 1% level" |
| 0.01 <= p < 0.05 | "Statistically significant at the 5% level" |
| 0.05 <= p < 0.10 | "Marginally significant" or "significant at the 10% level only" |
| p >= 0.10 | "Not statistically significant" or "insufficient evidence to reject the null" |

### What NOT to Say

- "The result is significant" without specifying the level.
- "The result proves that X causes Y." Statistical tests do not prove causation.
- "The result is insignificant, therefore the effect does not exist." Absence of evidence is not evidence of absence. The test may lack power.
- "The p-value is 0.06, which is close to significant." Either apply a pre-specified significance level or report the exact p-value and let the reader decide.
- "The model works." Specify what "works" means: adequate diagnostics, good out-of-sample fit, economically meaningful parameters.

### Statistical vs Economic Significance

A parameter can be statistically significant but economically irrelevant. For example:
- An AR(1) coefficient of 0.02 with p < 0.001 is statistically significant but implies negligible return predictability (R^2 ~ 0.04%).
- A GARCH alpha of 0.001 may be significant with enough data but contributes almost nothing to variance dynamics.

Always assess economic significance alongside statistical significance:
- "The leverage effect (gamma = 0.08, p < 0.001) implies that negative shocks have 2.1 times the volatility impact of positive shocks, which is economically meaningful for risk management."

## Uncertainty Quantification

### Always Report Confidence Intervals

Point estimates without uncertainty measures are incomplete. For every key quantity, report a confidence interval or standard error:

- "Unconditional annualized volatility is estimated at 17.1% (95% CI: 15.8% - 18.4%)."
- "The persistence parameter alpha + beta = 0.986 (SE: 0.005)."
- "Expected bear-regime duration is 8.3 months (bootstrap 90% CI: 5.1 - 14.2 months)."

### Parameter Uncertainty Propagation

When computing derived quantities (e.g., unconditional variance from omega, alpha, beta), uncertainty propagates. The delta method or bootstrap provides standard errors for nonlinear transformations:

- Unconditional variance = omega / (1 - alpha - beta). A small change in persistence (alpha + beta) near 1 causes large changes in unconditional variance. Report this sensitivity.

### Scenario Analysis

For forecasts and risk measures, complement point estimates with scenarios:

- **Base case**: Volatility reverts to its long-run average (17.1% annualized).
- **Stress case**: Volatility remains at current elevated levels (28% annualized) for 3 months before mean-reverting.
- **Tail case**: A 2008-magnitude shock occurs, pushing conditional volatility to 60%+ annualized.

## Visual Presentation Standards

### Required Plots

Every time-series report should include:

1. **Return series plot**: Raw returns over time with notable events annotated.
2. **Conditional volatility plot**: Estimated volatility over time, with crisis periods shaded.
3. **Diagnostic plots**: Residual ACF, squared residual ACF, QQ-plot.
4. **Forecast plot** (if applicable): Point forecasts with confidence bands.

### Plot Formatting Rules

- Label all axes with units (e.g., "Annualized Volatility (%)", not just "sigma").
- Include a descriptive title (e.g., "GJR-GARCH(1,1) Conditional Volatility: S&P 500 Daily Returns, 2005-2024").
- Use consistent date formatting on time axes.
- Include gridlines for readability.
- If comparing models, use distinct colors with a legend.
- Annotate key events (crisis dates, regime transitions) directly on the plot.

### Table Formatting Rules

- Right-align numerical columns.
- Use consistent decimal places (4 for parameter estimates, 3 for p-values, 1 for AIC/BIC).
- Bold or mark the best model in comparison tables.
- Include units in column headers.

## Tone and Claims

### Appropriate Claims

- "The GJR-GARCH(1,1) model provides a statistically adequate description of S&P 500 return volatility dynamics over the sample period, as evidenced by clean residual diagnostics."
- "Current conditional volatility (22.4% annualized) is elevated relative to the long-run estimate (17.1%), suggesting above-average risk."
- "The model identifies two periods consistent with high-volatility regimes: 2008-2009 and March-June 2020."

### Inappropriate Claims

- "The model predicts that the market will crash." (Models estimate conditional distributions, not specific outcomes.)
- "Our analysis proves that leverage effects drive equity volatility." (The model is consistent with leverage effects; it does not prove a causal mechanism.)
- "The forecast is accurate to within 2%." (Volatility forecasts have wide uncertainty bands; point precision claims are misleading.)
- "Based on our GARCH model, we recommend going long/short." (Volatility models inform risk estimates, not trading direction.)

### Hedging Language (Use Appropriately)

- "The results suggest..." (for interpretations beyond direct parameter estimates).
- "Consistent with..." (when results align with economic theory).
- "Subject to the assumption that..." (when a conclusion depends on a specific assumption).
- "Over the sample period..." (to delimit the scope of findings).

Do not over-hedge. If the evidence is strong, say so: "The data strongly reject the null of no ARCH effects (ARCH-LM p < 0.001)."

## Checklist Before Finalizing

Before issuing any report, verify:

- [ ] Executive summary contains a clear finding, method, and caveat.
- [ ] Data description is complete (source, period, frequency, transformations, N).
- [ ] Model specification is written out mathematically.
- [ ] Parameter estimates are reported with standard errors and p-values.
- [ ] Information criteria (AIC, BIC) are reported.
- [ ] All diagnostic tests are reported with p-values and pass/fail interpretation.
- [ ] Failed diagnostics are explicitly discussed, not hidden.
- [ ] Forecasts include confidence intervals or scenario bands.
- [ ] Caveats section addresses model assumptions, sample limitations, and forecast limits.
- [ ] No causal claims are made from correlational evidence.
- [ ] Statistical and economic significance are both assessed.
- [ ] All plots have labeled axes, titles, and legends.
