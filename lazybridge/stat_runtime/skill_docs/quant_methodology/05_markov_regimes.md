# Markov Regime-Switching Models

## What Regime-Switching Models Detect

Markov regime-switching (MRS) models, introduced by Hamilton (1989), posit that the data-generating process switches between a finite number of discrete **regimes** or **states**. Each regime has its own set of parameters (mean, variance, or both). The transitions between regimes are governed by a Markov chain with fixed transition probabilities.

The key insight: financial time series often exhibit distinct behavioral modes -- bull markets with low volatility, bear markets with high volatility, crisis periods with extreme movements. Rather than fitting a single model to the entire sample and getting averaged-out parameters, regime-switching models identify these modes and estimate separate parameters for each.

## Core Model Structure

### Two-Regime Model for Returns

The most common specification for financial returns:

```
r_t = mu_{s_t} + sigma_{s_t} * epsilon_t,   epsilon_t ~ N(0,1)
s_t in {0, 1}
```

- **Regime 0** (typically "low volatility" or "bull"): mean = `mu_0`, volatility = `sigma_0`
- **Regime 1** (typically "high volatility" or "bear"): mean = `mu_1`, volatility = `sigma_1`
- `s_t` is an unobserved (latent) state variable that follows a first-order Markov chain.

### Transition Matrix

The transition matrix `P` governs regime dynamics:

```
P = | p_00  p_01 |
    | p_10  p_11 |
```

- `p_00` = P(stay in regime 0 | currently in regime 0). Typical values for monthly equity data: 0.90-0.98.
- `p_11` = P(stay in regime 1 | currently in regime 1). Typical values: 0.80-0.95.
- `p_01 = 1 - p_00` = P(switch from regime 0 to regime 1).
- `p_10 = 1 - p_11` = P(switch from regime 1 to regime 0).

### Interpreting the Transition Matrix

**Expected regime duration**: The expected duration of regime j is `1 / (1 - p_jj)`.

- If `p_00 = 0.97`, expected bull market duration = 1/0.03 = 33.3 months (about 2.8 years).
- If `p_11 = 0.90`, expected bear market duration = 1/0.10 = 10 months.

**Ergodic (unconditional) probabilities**: The long-run probability of being in regime 0 is `pi_0 = (1 - p_11) / (2 - p_00 - p_11)`. With p_00=0.97, p_11=0.90: pi_0 = 0.10/0.13 = 0.77, meaning the market spends about 77% of the time in the low-volatility regime.

**Asymmetry in transitions**: Bear markets (high-vol regime) typically have shorter expected duration but higher exit probability per period than bull markets. This is consistent with the empirical observation that crashes are sharp and recoveries are gradual.

## Smoothed vs Filtered Probabilities

The model produces two types of regime probability estimates:

### Filtered Probabilities

`P(s_t = j | information up to time t)`

These are real-time estimates. At each point in time, they use only data available up to that point. Useful for simulating what an investor would have known in real time.

### Smoothed Probabilities

`P(s_t = j | full sample information)`

These use the entire sample (past and future data relative to time t) via the Kim smoother algorithm. They are less noisy and provide the best retrospective estimate of which regime prevailed at each point.

**When to use which**:
- Use **smoothed probabilities** for historical analysis, regime dating, and reporting ("the model identifies a high-volatility regime from March 2020 to June 2020").
- Use **filtered probabilities** for evaluating real-time signal quality and for any trading/allocation strategy backtest (to avoid look-ahead bias).
- A smoothed probability above 0.80 is generally considered strong evidence for that regime. Between 0.50 and 0.80 is ambiguous. Below 0.50 indicates the other regime is more likely.

## Bull/Bear Market Detection

### Typical Two-Regime Results for Equity Indices

For monthly S&P 500 returns over long samples (1950-present), expect approximately:

| Parameter | Regime 0 (Bull) | Regime 1 (Bear) |
|---|---|---|
| Mean (monthly) | +0.8% to +1.2% | -1.5% to -0.5% |
| Std dev (monthly) | 3.0% to 4.0% | 5.5% to 8.0% |
| Duration (months) | 25-40 | 5-15 |
| Ergodic probability | 0.70-0.85 | 0.15-0.30 |

For daily returns, the separation is typically weaker in mean but stronger in variance. Mean differences are often insignificant at daily frequency.

### Validating Regime Identification

Cross-check detected regimes against known events:
- Does the bear regime activate during NBER recessions?
- Does it capture major crises (2008 GFC, 2020 COVID, dot-com bust)?
- Are there false positives (regimes triggered by minor corrections)?

If the model places 2008 and 2020 firmly in the high-volatility regime (smoothed probability > 0.90) and normal periods firmly in the low-volatility regime, it is performing as expected.

### Switching vs Threshold Models

Markov switching models differ from threshold models (TAR, SETAR) in that regimes are **latent** -- you never observe the regime directly. Threshold models condition on an observable variable (e.g., "bear market when trailing 12-month return < -10%"). Markov switching is preferred when:
- You do not have a clear observable threshold variable.
- You want the data to endogenously determine regime boundaries.
- You believe regime transitions are probabilistic, not deterministic.

## Comparing 2 vs 3 Regime Models

### Two-Regime Model

The standard choice. Captures the primary bull/bear distinction. Advantages:
- Fewer parameters (easier to estimate, less prone to overfitting).
- Clearer economic interpretation.
- More robust with limited data (works with 200+ monthly observations).

### Three-Regime Model

Adds a third state, often interpreted as:
- **Regime 0**: Low volatility, positive mean (calm bull).
- **Regime 1**: Moderate volatility, near-zero mean (turbulent/transitional).
- **Regime 2**: High volatility, negative mean (crisis).

Or alternatively:
- **Regime 0**: Normal market (moderate vol, positive mean).
- **Regime 1**: Bull market (low vol, high positive mean).
- **Regime 2**: Bear/crisis (high vol, negative mean).

### When to Use 3 Regimes

Use a 3-regime model when:
- The 2-regime model shows poor fit in the middle of the distribution (e.g., the bear regime conflates mild corrections with severe crises).
- AIC or BIC favors the 3-regime model (but be cautious -- see below).
- You have a long sample (500+ observations for monthly data; 2000+ for daily).
- The third regime has a clear economic interpretation and is not just an artifact.

### When NOT to Use 3 Regimes

- With fewer than 300 monthly observations (15+ years). Three regimes with a 3x3 transition matrix means 6 free transition probabilities plus 6 regime-specific parameters = 12+ parameters. Estimation becomes unreliable.
- When the third regime captures only 1-2 episodes. This is overfitting to specific events, not identifying a recurring state.
- When the transition matrix has rows with probabilities very close to 0 or 1, indicating near-degenerate regimes.
- BIC penalty for extra parameters exceeds the log-likelihood improvement.

### Model Selection Between 2 and 3 Regimes

Standard likelihood ratio tests are NOT valid for regime-switching models because the null hypothesis (2 regimes) places the nuisance parameters (transition probabilities for the third regime) on the boundary of the parameter space. This violates the regularity conditions for the chi-squared distribution.

Instead, use:
1. **BIC comparison**: BIC with its stronger penalty is preferred over AIC for regime number selection.
2. **Cross-validation**: Fit on a training sample, evaluate log-likelihood on a holdout.
3. **Regime interpretability**: Can each regime be given a meaningful economic label? If the third regime lacks a clear interpretation, prefer 2 regimes.

## Model Extensions

### Markov-Switching GARCH

Combines regime switching with GARCH dynamics within each regime. The variance equation becomes regime-dependent:

```
sigma^2_t = omega_{s_t} + alpha_{s_t} * epsilon^2_{t-1} + beta_{s_t} * sigma^2_{t-1}
```

This is powerful but computationally expensive and can have identification issues. Use only with substantial data (1000+ daily observations).

### Markov-Switching with Exogenous Variables

Transition probabilities can depend on observed variables:

```
P(s_t = 1 | s_{t-1} = 0) = Logistic(a + b * spread_t)
```

For example, the probability of switching to a bear regime may increase when the yield curve inverts. This is called a **time-varying transition probability (TVTP)** model.

### Duration-Dependent Models

Standard Markov models have a geometric duration distribution (memoryless). Duration-dependent models allow the probability of leaving a regime to depend on how long you have been in it. Useful if you believe bear markets become more likely to end as they persist.

## Estimation Details

### Maximum Likelihood via EM Algorithm

Regime-switching models are typically estimated via the **Expectation-Maximization (EM) algorithm** or direct numerical optimization (Hamilton filter + BFGS). EM is more stable but can converge slowly.

### Starting Values Matter

The likelihood surface for regime-switching models is often multimodal. Results can be sensitive to initial parameter values. Best practice:
- Run estimation from multiple random starting points (at least 10-20).
- Compare log-likelihoods across runs. If they differ by more than 0.1, the global optimum may not have been found.
- Use economically motivated starting values (e.g., set initial regime means to the sample mean +/- 1 standard deviation).

### Label Switching

Regimes are not inherently ordered. The model may label the bull regime as "0" in one run and "1" in another. Always relabel regimes by a consistent criterion (e.g., order by variance: regime 0 = low variance, regime 1 = high variance).

## When NOT to Use Markov Switching

1. **When a simpler model suffices**: If GARCH captures volatility dynamics adequately and there is no evidence of discrete regime shifts, Markov switching adds unnecessary complexity.
2. **Small samples**: Fewer than 150 observations (for 2 regimes) makes estimation unreliable. Transition probabilities will be poorly identified.
3. **When regimes are observationally identified**: If you can define regimes by observable criteria (e.g., recession indicator, VIX level), simple dummy variable models or threshold models are more transparent.
4. **When you need precise timing**: Regime-switching models identify regimes with a lag. The filtered probability often does not cross 0.50 until well into a new regime. Do not expect real-time regime detection.
5. **Non-stationary data**: Regime-switching models assume stationarity within each regime and stationarity of the switching process. Trending data or structural breaks in transition probabilities violate these assumptions.
6. **Overinterpretation of short regimes**: If the model identifies very brief regimes (1-2 periods), treat them with skepticism. They may reflect outliers rather than genuine regime changes.

## Diagnostics for Regime-Switching Models

### Residual Checks

Compute regime-weighted residuals:
```
epsilon_t = r_t - sum_j P(s_t=j|full sample) * mu_j
```

These should be approximately white noise. Check with Ljung-Box test.

### Regime Classification Measure (RCM)

```
RCM = 1 - (1/T) * sum_t (2 * |P(s_t=0|full sample) - 0.5|)
```

RCM ranges from 0 (perfect classification -- all smoothed probabilities near 0 or 1) to 1 (no classification -- all probabilities near 0.5). Values below 0.30 indicate good regime separation. Values above 0.50 suggest the model struggles to distinguish regimes.

### Regime Stability

Check whether the regime classification is stable across sub-samples. Fit the model on the first half and second half of the sample separately. If regime parameters change dramatically, the regime structure may not be stable.

## Reporting Conventions

When reporting regime-switching results, include:

- Number of regimes and specification (switching mean, switching variance, or both)
- Regime-specific parameter estimates with standard errors
- Full transition matrix with expected durations for each regime
- Ergodic probabilities
- Time series plot of smoothed probabilities with regime shading
- Log-likelihood, AIC, BIC
- Comparison with non-switching baseline model
- Sample period and observation count
- Note on number of random starting values tried and whether convergence was consistent
- Regime Classification Measure (RCM)
