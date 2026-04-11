# Writing Disciplined Research Reports

## Report Structure

### 1. Executive Summary (2-3 sentences)
- What was analyzed, what was found, what it means
- Include the key metric (e.g., "GARCH(1,1) estimates annualized volatility at 18.5%")
- State the confidence level of findings

### 2. Data Description
- Dataset name, source, frequency, date range
- Number of observations
- Key descriptive statistics (mean, std, min, max)
- Any data quality issues (missing values, outliers)

### 3. Methodology
- Which model(s) were fitted and why
- Key parameters and their values
- Diagnostic tests applied
- Software/tools used

### 4. Results
- Parameter estimates with standard errors
- Information criteria (AIC, BIC)
- Key metrics (R², log-likelihood)
- Diagnostic test results
- Visualizations with descriptions

### 5. Interpretation
- What the parameters mean in economic terms
- What the diagnostics tell us about model adequacy
- Comparison with alternative specifications
- Caveats and limitations

### 6. Conclusion
- Summary of findings
- Actionable recommendations (if applicable)
- Suggested next steps

## Language Standards

### Do Say
- "The ADF test rejects the null hypothesis of a unit root at the 5% level (p = 0.003)"
- "The GARCH(1,1) model estimates conditional volatility ranging from 12% to 45% annualized"
- "The model explains approximately 65% of return variance (adjusted R² = 0.648)"
- "Results should be interpreted with caution given the limited sample size (n = 250)"

### Do NOT Say
- "The model proves that..." (models don't prove, they provide evidence)
- "The results are significant" (specify: statistically significant at what level)
- "This will happen" (forecasts are uncertain — use "the model projects" or "estimates suggest")
- "The best model is..." without specifying the criterion

### Uncertainty Language
| Confidence | Language |
|-----------|----------|
| Very high (p < 0.001) | "Strong evidence that..." |
| High (p < 0.01) | "Evidence supports..." |
| Moderate (p < 0.05) | "The data suggest..." |
| Weak (p < 0.10) | "Marginal evidence for..." |
| None (p > 0.10) | "No significant evidence of..." |

## Visualization Standards
- Every plot needs a title and axis labels
- Include units on axes (%, dollars, date)
- Confidence intervals shown as shaded bands
- Reference lines where appropriate (e.g., zero line for residuals)
- Multiple series clearly distinguished by color or style

## What NOT to Claim
1. Causality from correlation
2. Future certainty from historical patterns
3. Model validity beyond the estimation sample
4. Precision beyond what the data supports
5. Universal applicability of sample-specific results

## Caveats Section (Required)
Every report must acknowledge:
- Sample period limitations
- Model assumptions and where they may break
- Data quality issues
- Known structural changes in the analysis period
- Limitations of the chosen methodology
