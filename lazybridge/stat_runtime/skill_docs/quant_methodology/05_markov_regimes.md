# Markov Switching Regime Models

## What They Detect
Markov switching models identify hidden states (regimes) in time series data. In finance, these typically correspond to:
- Bull vs. bear markets
- High vs. low volatility periods
- Expansion vs. recession phases
- Risk-on vs. risk-off environments

## When to Use
- Data appears to switch between distinct behaviors
- Mean and/or variance change over time
- You suspect regime-dependent dynamics
- Structural breaks exist but timing is unknown
- Volatility regimes drive investment decisions

## When NOT to Use
- Data is stationary with constant parameters (use OLS/ARIMA)
- Changes are gradual (consider time-varying parameter models)
- Very short series (<200 observations) — insufficient for regime detection
- Too many potential regimes (>3 rarely justified statistically)

## Key Concepts

### Transition Matrix
For a 2-regime model:
```
P = | p₁₁  p₁₂ |
    | p₂₁  p₂₂ |
```
- p₁₁: probability of staying in regime 1
- p₁₂: probability of switching from regime 1 to regime 2
- Each row sums to 1

### Expected Regime Duration
Duration of regime i = 1 / (1 - p_ii)
- If p₁₁ = 0.98: expected duration = 50 periods
- If p₂₂ = 0.95: expected duration = 20 periods
- Persistent regimes (high diagonal) are more meaningful

### Smoothed Probabilities
- Probability of being in each regime at each time point
- Uses full sample information (backward-looking)
- Values near 0 or 1 indicate clear regime assignment
- Values near 0.5 indicate uncertainty about the regime

## Interpretation Guide

### 2-Regime Model (Standard)
- **Regime 0**: Typically low-volatility / normal market
- **Regime 1**: Typically high-volatility / stressed market
- Check: regime means should differ meaningfully
- Check: smoothed probabilities should show clear separation

### 3-Regime Model
- Adds a third state (e.g., transition/moderate regime)
- Only use if AIC/BIC clearly improves over 2-regime
- Risk: overfitting with small samples

### Regime Classification Certainty
Average max smoothed probability across all observations:
- \> 0.90: Excellent regime separation
- 0.70-0.90: Good separation
- < 0.70: Weak — regimes may not be distinct

## Model Selection
1. Start with 2 regimes (most common and interpretable)
2. Compare 2 vs 3 regimes using BIC (penalizes complexity more)
3. Check if additional regimes have economic meaning
4. Verify regime durations are reasonable (not 1-2 periods)

## Forecasting with Markov Models
- Forecasts are regime-probability weighted averages
- Short-term: current regime dominates
- Long-term: converges to ergodic (stationary) probabilities
- Forecast uncertainty is higher than single-regime models

## Common Pitfalls
1. **Too many regimes**: 2 is usually sufficient, 3 occasionally justified
2. **Ignoring economic meaning**: Regimes should correspond to interpretable states
3. **Short regime durations**: Expected duration < 5 periods suggests noise, not regimes
4. **Non-convergence**: Try different starting values; simplify the model
5. **Overinterpreting transitions**: Not every switch is a structural change
6. **Comparing with GARCH**: Markov models regime means, GARCH models regime variance — they complement each other
