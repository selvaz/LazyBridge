# Plots and Visualization Reference

## Overview

The statistical runtime automatically generates plots when `fit_model` succeeds. Plots are saved as PNG files (150 DPI) in the artifact store and registered in the MetaStore. You do not need to call a separate tool to generate plots -- they are produced as part of the fit pipeline. Use `get_plot` and `list_artifacts` to retrieve plot file paths after fitting.

All plots use matplotlib with a consistent style: 10x6 inch figures, grid enabled, 11pt font, non-interactive Agg backend (headless-safe).

## Plot Catalog

### Plots Generated for All Model Families

| Plot Name | File Name | Description | When Generated |
|---|---|---|---|
| `residuals` | `residuals.png` | Two-panel: residual scatter plot (left) + residual histogram (right) | Always, if residuals exist |
| `acf_pacf` | `acf_pacf.png` | Side-by-side ACF and PACF correlograms of residuals | Always, if residuals exist |
| `forecast` | `forecast.png` | Observed data (last 100 points) + forecast with confidence band | Only if `forecast_steps` was set |

### Family-Specific Plots

| Plot Name | File Name | Family | Description | When Generated |
|---|---|---|---|---|
| `volatility` | `volatility.png` | GARCH | Conditional volatility (crimson line) overlaid on returns (gray) using dual y-axes | Always for GARCH fits |
| `regimes` | `regimes.png` | Markov | Multi-panel: observed series (top) + smoothed probability for each regime (below) | Always for Markov fits |

### On-Demand Plots (Not Auto-Generated)

| Plot Name | File Name | Description | How to Generate |
|---|---|---|---|
| `model_comparison` | `model_comparison.png` | AIC/BIC bar chart comparing multiple runs | Call `compare_models(run_ids)`, then check artifacts |
| `series` | `series.png` | Basic time series line plot | Not available via tools directly; requires programmatic access |

## Retrieving Plots

### Method 1: get_plot Tool

```python
get_plot(run_id="a1b2c3d4e5f6g7h8", name="residuals")
```

Returns:

```json
{
  "path": "/absolute/path/artifacts/a1b2c3d4e5f6g7h8/plots/residuals.png",
  "name": "residuals",
  "description": "Residual scatter plot and histogram"
}
```

If the plot name does not exist:

```json
{
  "error": true,
  "message": "Plot 'volatility' not found. Available: ['residuals', 'acf_pacf', 'forecast']"
}
```

The error message lists all available plot names for that run, which is useful for discovering what was generated.

### Method 2: list_artifacts Tool

```python
list_artifacts(run_id="a1b2c3d4e5f6g7h8", artifact_type="plot")
```

Returns a list of all plot artifacts:

```json
[
  {
    "run_id": "a1b2c3d4e5f6g7h8",
    "name": "residuals",
    "artifact_type": "plot",
    "file_format": "png",
    "path": "/absolute/path/artifacts/a1b2c3d4e5f6g7h8/plots/residuals.png",
    "description": "Residual scatter plot and histogram"
  },
  {
    "run_id": "a1b2c3d4e5f6g7h8",
    "name": "acf_pacf",
    "artifact_type": "plot",
    "file_format": "png",
    "path": "/absolute/path/artifacts/a1b2c3d4e5f6g7h8/plots/acf_pacf.png",
    "description": "ACF and PACF correlogram"
  }
]
```

### Method 3: Check artifact_paths in RunRecord

The `fit_model` return value includes `artifact_paths`, which lists all generated file paths (plots, data, summaries). Filter by `/plots/` in the path to find plot files.

## Artifact Storage Layout

```
{artifacts_dir}/
    {run_id}/
        plots/
            residuals.png
            acf_pacf.png
            volatility.png      # GARCH only
            regimes.png         # Markov only
            forecast.png        # If forecast_steps was set
            model_comparison.png # If compare_models was called
        data/
            residuals.json
            forecast.json       # If forecast was generated
        summaries/
            spec.json
            fit_summary.txt
            diagnostics.json
```

The default `artifacts_dir` is `"artifacts"` (relative to the working directory). It is set when creating the `StatRuntime`.

## Valid Plot Names for get_plot

Use these exact strings as the `name` parameter:

| Name String | When Available |
|---|---|
| `"residuals"` | All successful fits |
| `"acf_pacf"` | All successful fits |
| `"forecast"` | Fits with `forecast_steps > 0` |
| `"volatility"` | GARCH fits only |
| `"regimes"` | Markov fits only |
| `"model_comparison"` | After `compare_models` is called |
| `"series"` | Only if explicitly generated programmatically |

## Plot Details

### residuals Plot

Two panels side by side:
- **Left panel**: Scatter plot of residuals vs observation index. A red dashed horizontal line at y=0. Marker size 2, alpha 0.6.
- **Right panel**: Histogram of residual values. Bin count = min(50, n/5 + 1). Black edges, alpha 0.7.

**Interpretation guidance**: Look for patterns in the scatter plot (should be random). The histogram should be roughly bell-shaped for well-specified models. Asymmetry or heavy tails indicate non-normality.

### acf_pacf Plot

Two panels side by side:
- **Left panel**: Autocorrelation Function (ACF) up to min(40, n/2 - 1) lags with 95% confidence bands.
- **Right panel**: Partial Autocorrelation Function (PACF) using the Yule-Walker method ("ywm"), same lag range.

Both use statsmodels `plot_acf` and `plot_pacf` with `alpha=0.05` (95% significance bands).

**Interpretation guidance**: Significant spikes outside the confidence bands indicate remaining autocorrelation. For a well-fitted model, residual ACF/PACF should show no significant spikes.

### volatility Plot (GARCH only)

Dual-axis plot:
- **Left y-axis** (gray): Raw returns series, plotted as a thin gray line.
- **Right y-axis** (crimson): Conditional volatility estimate from the GARCH model.

Figure size: 12x6 inches.

**Interpretation guidance**: Volatility should cluster around market events. Periods of high volatility should correspond to turbulent market conditions.

### regimes Plot (Markov only)

Multi-panel vertical stack with shared x-axis:
- **Top panel** (if data is available): Observed series in black.
- **Subsequent panels**: One per regime, showing the smoothed probability (0 to 1) as a filled area plot.

Colors cycle through: `#2196F3` (blue), `#FF5722` (red-orange), `#4CAF50` (green), `#FF9800` (amber).

Each panel is 12 inches wide and 3 inches tall per sub-panel.

**Interpretation guidance**: When a regime's probability is near 1.0, the model is confident the system is in that regime. Look for clean regime transitions (probabilities switching sharply between 0 and 1) as evidence of good regime separation.

### forecast Plot

Single-panel plot:
- **Black line**: Last 100 observations of actual data (configurable via `last_n` parameter internally, default 100).
- **Blue line**: Point forecast beyond the end of actual data.
- **Blue shaded region**: Confidence interval (default 95%).

Figure size: 12x6 inches.

**Interpretation guidance**: Width of the CI band reflects forecast uncertainty. For ARIMA, CIs typically widen as the horizon increases. For GARCH, CIs reflect volatility uncertainty.

### model_comparison Plot

Bar chart:
- **Blue bars**: AIC for each model run.
- **Red-orange bars**: BIC for each model run.
- **X-axis labels**: Engine name + truncated run ID.
- Value labels above each bar.

Figure width scales with the number of models: max(8, n_models * 2) inches.

## Plot Generation Failure

Plot generation is non-fatal. If matplotlib fails or data is incompatible with a plot type, the fit still succeeds. A warning is logged but the `RunRecord` status remains `"success"`. The plot simply will not appear in `artifact_paths` or the artifact store.

Check if a specific plot was generated:

```python
result = get_plot(run_id="...", name="volatility")
if "error" in result:
    # Plot was not generated for this run
    pass
```

Or list all available plots:

```python
plots = list_artifacts(run_id="...", artifact_type="plot")
available_names = [p["name"] for p in plots]
```

## Expected Plots by Family

| Family | residuals | acf_pacf | forecast | volatility | regimes |
|---|---|---|---|---|---|
| OLS | Yes | Yes | If forecast_steps set | No | No |
| ARIMA | Yes | Yes | If forecast_steps set | No | No |
| GARCH | Yes | Yes | If forecast_steps set | Yes | No |
| Markov | Yes | Yes | If forecast_steps set | No | Yes |

## Non-Plot Artifacts

The `list_artifacts` tool also returns non-plot artifacts when called without `artifact_type` or with other types:

| Type | Name | Format | Description |
|---|---|---|---|
| `summary` | `spec` | JSON | The `ModelSpec` used for this run |
| `summary` | `fit_summary` | TXT | Full model summary text (statsmodels/arch output) |
| `summary` | `diagnostics` | JSON | All diagnostic test results |
| `data` | `residuals` | JSON | Raw residual values as a list of floats |
| `forecast` | `forecast` | JSON | Full `ForecastResult` dict (point forecast, CIs, extras) |

### Retrieving Non-Plot Artifacts

```python
# List all artifacts
all_artifacts = list_artifacts(run_id="...", artifact_type=None)

# List only data artifacts
data_artifacts = list_artifacts(run_id="...", artifact_type="data")

# List only summary artifacts
summaries = list_artifacts(run_id="...", artifact_type="summary")
```

## Workflow: Get All Plots After Fitting

```python
# Step 1: Fit the model
result = fit_model(
    family="garch",
    target_col="ret",
    dataset_name="equities",
    params={"p": 1, "q": 1},
    forecast_steps=20
)
run_id = result["run_id"]

# Step 2: List all plots
plots = list_artifacts(run_id=run_id, artifact_type="plot")
# Expected: residuals, acf_pacf, volatility, forecast

# Step 3: Get a specific plot path
vol_plot = get_plot(run_id=run_id, name="volatility")
print(vol_plot["path"])  # /absolute/path/to/volatility.png

# Step 4: Get the forecast plot
fc_plot = get_plot(run_id=run_id, name="forecast")
print(fc_plot["path"])  # /absolute/path/to/forecast.png
```

## Troubleshooting Missing Plots

| Symptom | Cause | Fix |
|---|---|---|
| No `volatility` plot | Not a GARCH model | Volatility plot is GARCH-only |
| No `regimes` plot | Not a Markov model | Regimes plot is Markov-only |
| No `forecast` plot | `forecast_steps` was `None` or `0` | Re-fit with `forecast_steps=N` |
| No plots at all | `matplotlib` not installed | Run `pip install lazybridge[stats]` |
| No `acf_pacf` plot | `statsmodels` not installed or residuals empty | Check installation and data |
| `get_plot` returns error | Plot name misspelled | Check error message for available names |
