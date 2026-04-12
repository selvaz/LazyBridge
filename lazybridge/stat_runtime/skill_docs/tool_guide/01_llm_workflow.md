# LLM Agent Workflow Guide

This is the primary reference for how an LLM agent should use the statistical runtime. Follow this workflow for every analysis request.

## Default Workflow

```
1. discover_data()              → What data is available?
2. Read the summary + column roles → Pick dataset and target
3. analyze(dataset, target)     → Run analysis (mode="recommend" by default)
4. Report the narrative + artifacts
5. Only use expert tools if the user asks for custom control
```

## When to Use Each Mode

| User says | Mode | What happens |
|---|---|---|
| "analyze", "look at", "what's going on" | `recommend` | Runtime inspects data and picks best model |
| "forecast", "predict", "project" | `forecast` | ARIMA time-series forecast |
| "volatility", "risk", "VaR" | `volatility` | GARCH volatility modeling |
| "regime", "bull/bear", "switching" | `regime` | Markov regime detection |
| "describe", "summarize", "overview" | `describe` | Descriptive stats + stationarity tests |

## Auto-Detection

- **Time column**: auto-detected from dataset metadata if not specified
- **Model family**: `mode="recommend"` inspects column roles and data size to choose
- **Forecast steps**: auto-set to 20 for forecast/volatility modes if not specified
- **Entity filtering**: use `group_col` + `group_value` to filter panel data (e.g., `group_col="symbol", group_value="SPY"`)

## When to Fall Back to Expert Tools

Only use low-level tools (`fit_model`, `query_data`, etc.) when:
- User wants specific ARIMA(2,1,3) parameters
- User wants custom SQL joins or computed columns
- User wants to compare specific run IDs manually
- User wants a specific plot regenerated
- User wants to run stationarity tests independently

## Common Mistakes to Avoid

1. **Hallucinated columns**: Always call `discover_data()` first and use only columns that appear in the result
2. **IDs as regressors**: Check `identifiers_to_ignore` and column roles — never use identifier columns in modeling
3. **Forecasting without time**: If no time column exists, forecast mode will fail — use describe instead
4. **Categorical as continuous**: String/categorical columns cannot be targets for ARIMA/GARCH — only numeric columns
5. **Prices instead of returns**: GARCH expects returns (centered near 0), not price levels — check the column role inference

## Reading the Results

The `analyze()` result includes:
- `mode_rationale`: Why this analysis was chosen
- `assumptions`: What the model assumes about your data
- `interpretation`: What the results mean in plain English
- `model_adequate`: Whether diagnostics pass (True = trustworthy)
- `warnings`: Things to watch out for
- `next_steps`: What to try next
- `plots`: Available visualization artifacts with file paths

## Two-Tier Architecture

- **High-level tools** (main agent): `discover_data`, `discover_analyses`, `analyze`, `register_dataset`
- **Expert tools** (sub-agent): `fit_model`, `forecast_model`, `query_data`, `profile_dataset`, `run_diagnostics`, `compare_models`, `get_run`, `list_runs`, `list_datasets`, `list_artifacts`, `get_plot`

Use `delegate_to_expert()` to access expert tools when the main agent has only high-level tools loaded.
