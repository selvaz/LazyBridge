# Artifact Management Reference

## What Artifacts Are

Artifacts are files stored on disk per model run. Every successful `fit_model` call produces a set of artifacts: plots (PNG images), data exports (JSON), and summaries (JSON/TXT). These files are organized by run ID in a directory tree under the runtime's `artifacts_dir` root (default: `./artifacts`).

Each artifact is also registered in the `MetaStore` as an `ArtifactRecord`, making them queryable via the `list_artifacts` and `get_plot` tools without knowing the filesystem layout.

## Directory Layout

```
{artifacts_dir}/
    {run_id}/
        plots/
            residuals.png
            acf_pacf.png
            volatility.png       # GARCH only
            regimes.png          # Markov only
            forecast.png         # Only if forecast_steps was set
            model_comparison.png # Only from compare_models
        data/
            residuals.json
            forecast.json        # Only if forecast_steps was set
        summaries/
            spec.json
            fit_summary.txt
            diagnostics.json
```

The `{run_id}` is the 16-character hex ID returned by `fit_model`. Subdirectories (`plots/`, `data/`, `summaries/`) are created automatically. The mapping from artifact type to subdirectory is:

| Artifact Type | Subdirectory |
|---|---|
| `plot` | `plots/` |
| `data` | `data/` |
| `summary` | `summaries/` |
| `forecast` | `data/` |

Note that `forecast` type artifacts are stored in the `data/` subdirectory, not a separate `forecast/` directory.

## ArtifactRecord Schema

Each artifact registered in the MetaStore has the following fields:

| Field | Type | Description |
|---|---|---|
| `run_id` | `str` | The run ID this artifact belongs to |
| `name` | `str` | Short name for the artifact (e.g. `"residuals"`, `"spec"`, `"forecast"`) |
| `artifact_type` | `str` | One of: `"plot"`, `"data"`, `"summary"`, `"forecast"` |
| `file_format` | `str` | File format: `"png"`, `"json"`, `"txt"`, `"csv"`, `"parquet"`, `"svg"` |
| `path` | `str` | Absolute filesystem path to the artifact file |
| `description` | `str` | Human-readable description (e.g. `"Residual scatter plot and histogram"`) |
| `created_at` | `datetime` | UTC timestamp when the artifact was created |
| `metadata` | `dict` | Optional additional metadata (usually empty) |

The primary key in the DuckDB backend is `(run_id, name)`. Saving an artifact with the same `run_id` and `name` overwrites the previous record.

## Tools for Artifact Access

### list_artifacts

Lists all artifacts for a given run, optionally filtered by type.

```python
list_artifacts(
    run_id: str,                        # Required
    artifact_type: str | None = None,   # Optional filter
)
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `run_id` | `str` | Yes | -- | The run ID to list artifacts for |
| `artifact_type` | `str \| None` | No | `None` | Filter by type. One of: `"plot"`, `"data"`, `"summary"`, `"forecast"`. `None` returns all types. |

**Returns**: A list of `ArtifactRecord` dicts.

```json
[
  {
    "run_id": "a1b2c3d4e5f6g7h8",
    "name": "residuals",
    "artifact_type": "plot",
    "file_format": "png",
    "path": "/abs/path/artifacts/a1b2c3d4e5f6g7h8/plots/residuals.png",
    "description": "Residual scatter plot and histogram",
    "created_at": "2025-01-15T10:31:00+00:00",
    "metadata": {}
  },
  {
    "run_id": "a1b2c3d4e5f6g7h8",
    "name": "acf_pacf",
    "artifact_type": "plot",
    "file_format": "png",
    "path": "/abs/path/artifacts/a1b2c3d4e5f6g7h8/plots/acf_pacf.png",
    "description": "ACF and PACF correlogram",
    "created_at": "2025-01-15T10:31:00+00:00",
    "metadata": {}
  }
]
```

**Examples**:

```python
# All artifacts for a run
list_artifacts(run_id="a1b2c3d4e5f6g7h8")

# Only plots
list_artifacts(run_id="a1b2c3d4e5f6g7h8", artifact_type="plot")

# Only summaries
list_artifacts(run_id="a1b2c3d4e5f6g7h8", artifact_type="summary")

# Only data exports
list_artifacts(run_id="a1b2c3d4e5f6g7h8", artifact_type="data")
```

### get_plot

Retrieves the file path and metadata for a specific named plot.

```python
get_plot(
    run_id: str,    # Required
    name: str,      # Required. Plot name (e.g. "residuals", "volatility")
)
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `run_id` | `str` | Yes | -- | The run ID |
| `name` | `str` | Yes | -- | The plot name. Must match the `name` field of a plot artifact. |

**Returns on success**:

```json
{
  "path": "/abs/path/artifacts/a1b2.../plots/residuals.png",
  "name": "residuals",
  "description": "Residual scatter plot and histogram"
}
```

**Returns on failure** (plot name not found):

```json
{
  "error": true,
  "message": "Plot 'volatility' not found. Available: ['residuals', 'acf_pacf', 'forecast']"
}
```

The error message lists all available plot names for the run, which is helpful for discovering what plots exist.

**Examples**:

```python
get_plot(run_id="a1b2c3d4e5f6g7h8", name="residuals")
get_plot(run_id="a1b2c3d4e5f6g7h8", name="volatility")
get_plot(run_id="a1b2c3d4e5f6g7h8", name="forecast")
get_plot(run_id="a1b2c3d4e5f6g7h8", name="regimes")
get_plot(run_id="a1b2c3d4e5f6g7h8", name="acf_pacf")
get_plot(run_id="a1b2c3d4e5f6g7h8", name="model_comparison")
```

## Artifact Types in Detail

### Plot Artifacts (artifact_type: "plot")

Plot artifacts are PNG images (150 DPI, tight bounding box) generated by matplotlib. They are saved via `ArtifactStore.write_plot()`, which calls `fig.savefig()` and then closes the figure to free memory.

#### Common Plot Names by Model Family

| Plot Name | Families | Description |
|---|---|---|
| `residuals` | ALL (OLS, ARIMA, GARCH, Markov) | Two-panel plot: residual scatter over time (left) + residual histogram (right) |
| `acf_pacf` | ALL | Two-panel plot: ACF (left) + PACF (right) of residuals, with 95% confidence bands. Up to 40 lags. |
| `volatility` | GARCH only | Dual-axis plot: returns (gray, left axis) + conditional volatility (red, right axis) |
| `regimes` | Markov only | Multi-panel plot: observed series (top panel) + smoothed probability for each regime (one panel per regime) |
| `forecast` | ALL (only if `forecast_steps` was set) | Last 100 actuals (black) + point forecast (blue) + CI bands (shaded blue). Description includes step count and CI level. |
| `model_comparison` | Generated by compare_models, not fit_model | Grouped bar chart: AIC (blue bars) and BIC (orange bars) for each model run. Value labels on each bar. |

### Data Artifacts (artifact_type: "data")

Data artifacts are JSON files containing numeric arrays or structured data.

| Name | Generated When | Contents |
|---|---|---|
| `residuals` | Always (if model produces residuals) | JSON array of floats: the model residuals. E.g. `[0.002, -0.001, 0.004, ...]` |
| `forecast` | Only if `forecast_steps` was set in `fit_model` | Full `ForecastResult` dict serialized as JSON, including `point_forecast`, `lower_ci`, `upper_ci`, `ci_level`, `dates`, `extra` |

Forecast data artifacts have `artifact_type: "forecast"` but are stored in the `data/` subdirectory.

### Summary Artifacts (artifact_type: "summary")

Summary artifacts capture the model specification, fit output, and diagnostic results.

| Name | Format | Contents |
|---|---|---|
| `spec` | JSON | The `ModelSpec` dict: `family`, `target_col`, `dataset_name`, `query_sql`, `exog_cols`, `params`, `forecast_steps`, `time_col` |
| `fit_summary` | TXT | The statsmodels/arch summary text output (the formatted table you see in a notebook). For GARCH, this is the arch library summary. |
| `diagnostics` | JSON | List of `DiagnosticResult` dicts, each with `test_name`, `statistic`, `p_value`, `passed`, `detail`, `interpretation` |

## How to Retrieve a Specific Artifact Path

### By Name (Recommended for Plots)

```python
result = get_plot(run_id="abc123", name="residuals")
if "error" not in result:
    plot_path = result["path"]
    # plot_path is an absolute filesystem path like "/home/user/artifacts/abc123/plots/residuals.png"
```

### By Listing and Filtering

```python
# Get all data artifacts
artifacts = list_artifacts(run_id="abc123", artifact_type="data")

# Find the residuals data
for a in artifacts:
    if a["name"] == "residuals":
        residuals_path = a["path"]
        break
```

### By Convention (Constructing the Path)

If you know the run_id, artifact name, and type, you can construct the path:

```
{artifacts_dir}/{run_id}/{type_dir}/{name}.{ext}
```

For example:
- `artifacts/abc123def456gh78/plots/residuals.png`
- `artifacts/abc123def456gh78/data/residuals.json`
- `artifacts/abc123def456gh78/summaries/spec.json`
- `artifacts/abc123def456gh78/summaries/fit_summary.txt`
- `artifacts/abc123def456gh78/summaries/diagnostics.json`

However, using `list_artifacts` or `get_plot` is preferred because they return the absolute path.

## Complete Artifact Inventory by Family

### OLS Artifacts

| Name | Type | Format | Always Present |
|---|---|---|---|
| `residuals` (plot) | plot | png | Yes |
| `acf_pacf` (plot) | plot | png | Yes |
| `forecast` (plot) | plot | png | Only if `forecast_steps` set |
| `residuals` (data) | data | json | Yes |
| `forecast` (data) | forecast | json | Only if `forecast_steps` set |
| `spec` | summary | json | Yes |
| `fit_summary` | summary | txt | Yes |
| `diagnostics` | summary | json | Yes |

### ARIMA Artifacts

Same as OLS.

### GARCH Artifacts

All OLS artifacts plus:

| Name | Type | Format | Always Present |
|---|---|---|---|
| `volatility` | plot | png | Yes (if conditional_volatility is available) |

### Markov Artifacts

All OLS artifacts plus:

| Name | Type | Format | Always Present |
|---|---|---|---|
| `regimes` | plot | png | Yes (if smoothed_probabilities is available) |

## Examples

### List All Plots for a GARCH Run

```python
plots = list_artifacts(run_id="abc123", artifact_type="plot")
# Returns: residuals, acf_pacf, volatility, and possibly forecast
```

### Get the Volatility Plot

```python
result = get_plot(run_id="abc123", name="volatility")
# {"path": "/abs/.../plots/volatility.png", "name": "volatility", "description": "GARCH conditional volatility plot"}
```

### Get the Forecast Data

```python
artifacts = list_artifacts(run_id="abc123", artifact_type="forecast")
# Returns the forecast data artifact with path to forecast.json
```

### Get the Model Specification

```python
summaries = list_artifacts(run_id="abc123", artifact_type="summary")
for s in summaries:
    if s["name"] == "spec":
        spec_path = s["path"]
        # Read spec_path to see the exact ModelSpec used
```

### Check What Plots Are Available Before Requesting

```python
# If unsure what plots exist, call get_plot with any name -- the error lists available plots
result = get_plot(run_id="abc123", name="anything")
# {"error": true, "message": "Plot 'anything' not found. Available: ['residuals', 'acf_pacf', 'volatility', 'forecast']"}
```

This pattern is useful for discovering which plots were generated for a run.
