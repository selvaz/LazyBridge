# Getting Started with the Statistical Runtime

## Overview

The statistical runtime (`stat_runtime`) provides tools for registering datasets, querying data with SQL, fitting statistical models (OLS, ARIMA, GARCH, Markov Switching), running diagnostics, generating plots, and persisting results. All tools are exposed as `LazyTool` instances that return plain dicts (JSON-serializable). Errors are never raised to the caller; they are returned as `{"error": true, "type": "...", "message": "..."}`.

## Installation

The stats extras must be installed:

```
pip install lazybridge[stats]
```

This installs: `duckdb`, `polars`, `sqlglot`, `statsmodels`, `arch`, `matplotlib`, `scipy`.

If any dependency is missing, tool calls will return an `ImportError` with the message `"... is required for this feature. Run: pip install lazybridge[stats]"`.

## Initializing the Runtime

Before any tool can be called, the runtime must be created and the tools bound to it.

### Minimal Setup (In-Memory)

```python
from lazybridge.stat_runtime.runner import StatRuntime
from lazybridge.stat_runtime.tools import stat_tools

rt = StatRuntime()
tools = stat_tools(rt)
```

This creates an in-memory runtime. All metadata (runs, artifacts, datasets) is stored in memory only and lost when the process exits.

### Persistent Setup (DuckDB + Disk Artifacts)

```python
rt = StatRuntime(db="my_project.duckdb", artifacts_dir="artifacts")
tools = stat_tools(rt)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `db` | `str \| None` | `None` | Path to a DuckDB database file for metadata persistence. `None` = in-memory. |
| `artifacts_dir` | `str` | `"artifacts"` | Root directory for plot PNGs, data exports, and summaries. |

### Context Manager Usage

```python
with StatRuntime(db="my.duckdb", artifacts_dir="out") as rt:
    tools = stat_tools(rt)
    # ... use tools ...
# rt.close() called automatically
```

### Binding to a LazyAgent

```python
from lazybridge import LazyAgent
from lazybridge.stat_runtime.runner import StatRuntime
from lazybridge.stat_runtime.tools import stat_tools

rt = StatRuntime()
agent = LazyAgent("anthropic", tools=stat_tools(rt))
resp = agent.loop("Register data.parquet, fit GARCH(1,1) on returns")
```

## Available Tools

After calling `stat_tools(rt)`, the following tools are available:

| Tool Name | Purpose |
|---|---|
| `register_dataset` | Register a Parquet or CSV file as a named dataset |
| `list_datasets` | List all registered datasets with schema and row count |
| `profile_dataset` | Compute column-level statistics (nulls, min, max, mean, std) |
| `query_data` | Execute a SQL SELECT query using `dataset('name')` macro |
| `fit_model` | Fit a statistical model (OLS, ARIMA, GARCH, Markov) |
| `forecast_model` | Generate a forecast from a previously fitted model |
| `run_diagnostics` | Run stationarity tests (ADF + KPSS) on a data column |
| `get_run` | Retrieve a past model run with metrics and artifact paths |
| `list_runs` | List past model runs, optionally filtered by dataset |
| `compare_models` | Compare multiple model runs by AIC, BIC, and other metrics |
| `list_artifacts` | List all artifacts (plots, data, summaries) for a model run |
| `get_plot` | Get the file path for a specific plot from a model run |

## Step 1: Register a Dataset

Every dataset must be registered before it can be used. Registration scans the file for schema and row count but does NOT load the data into memory.

### Parquet File

```python
register_dataset(
    name="equities",
    uri="/data/returns.parquet",
    time_column="date",
    frequency="daily",
    entity_keys=["symbol"]
)
```

### CSV File

```python
register_dataset(
    name="macro_data",
    uri="/data/gdp_quarterly.csv",
    time_column="quarter",
    frequency="quarterly"
)
```

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `str` | Yes | - | Logical name for the dataset (e.g. `"equities.daily"`) |
| `uri` | `str` | Yes | - | File path to a Parquet or CSV file |
| `time_column` | `str \| None` | No | `None` | Primary time/date column name |
| `frequency` | `str` | No | `"daily"` | One of: `daily`, `weekly`, `monthly`, `quarterly`, `annual`, `intraday`, `irregular` |
| `entity_keys` | `list[str] \| None` | No | `None` | Key columns for panel data (e.g. `["symbol", "country"]`) |

### Return Value

On success, returns the full `DatasetMeta` as a dict:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "name": "equities",
  "version": "1",
  "uri": "/data/returns.parquet",
  "file_format": "parquet",
  "schema_json": {"date": "Date", "symbol": "Utf8", "ret": "Float64", "volume": "Int64"},
  "frequency": "daily",
  "time_column": "date",
  "entity_keys": ["symbol"],
  "row_count": 125000,
  "registered_at": "2025-01-15T10:30:00+00:00"
}
```

On error (file not found, bad time_column):

```json
{"error": true, "type": "FileNotFoundError", "message": "Parquet path does not exist: /data/returns.parquet"}
```

## Step 2: Explore the Data

### List Datasets

```python
list_datasets()
```

Returns a list of dicts, each with: `name`, `uri`, `format`, `columns`, `row_count`, `time_column`, `frequency`.

### Profile a Dataset

```python
profile_dataset(name="equities")
```

Returns per-column stats: `dtype`, `null_count`, `null_pct`, `unique_count`, `min_val`, `max_val`, `mean`, `std`.

### Query with SQL

```python
query_data(
    sql="SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY' ORDER BY date",
    max_rows=5000
)
```

Rules:
- Only SELECT statements are allowed (INSERT, UPDATE, DELETE, DROP are blocked).
- Reference registered datasets with the `dataset('name')` macro.
- `max_rows` defaults to 5000. The result includes a `truncated` flag.
- SQL is executed via DuckDB -- standard SQL syntax applies.

### Run Stationarity Tests

Before fitting time-series models (ARIMA, GARCH), check stationarity:

```python
run_diagnostics(series_name="equities", column="ret")
```

Returns a list of two `DiagnosticResult` dicts (ADF and KPSS tests), each with `test_name`, `statistic`, `p_value`, `passed`, and `interpretation`.

## Step 3: Fit Your First Model

```python
result = fit_model(
    family="garch",
    target_col="ret",
    dataset_name="equities",
    params={"p": 1, "q": 1},
    forecast_steps=20
)
```

The `fit_model` tool:
1. Loads data from the registered dataset (or from a SQL query via `query_sql`).
2. Fits the model using the appropriate engine.
3. Runs automatic diagnostics (Ljung-Box, Jarque-Bera, etc.).
4. Generates plots (residuals, ACF/PACF, plus family-specific plots).
5. Generates a forecast if `forecast_steps` is set.
6. Persists everything: run record, metrics, artifacts.

### Return Value

The return value is a `RunRecord` dict containing:

```json
{
  "run_id": "a1b2c3d4e5f6g7h8",
  "dataset_name": "equities",
  "engine": "garch",
  "status": "success",
  "fit_summary": "... model summary text ...",
  "params_json": {"mu": 0.05, "omega": 0.01, "alpha[1]": 0.08, "beta[1]": 0.90},
  "metrics_json": {"aic": 1234.56, "bic": 1267.89, "log_likelihood": -612.28},
  "diagnostics_json": [...],
  "artifact_paths": ["artifacts/a1b2.../plots/residuals.png", ...],
  "duration_secs": 2.341,
  "created_at": "2025-01-15T10:31:00+00:00",
  "error_message": null
}
```

If the fit fails, `status` will be `"failed"` and `error_message` will contain the exception details. The tool never raises.

## Step 4: Retrieve Results

### Get a Run

```python
get_run(run_id="a1b2c3d4e5f6g7h8")
```

### List Past Runs

```python
list_runs(dataset_name="equities", limit=10)
```

### List Artifacts for a Run

```python
list_artifacts(run_id="a1b2c3d4e5f6g7h8", artifact_type="plot")
```

`artifact_type` can be: `"plot"`, `"data"`, `"summary"`, `"forecast"`.

### Get a Specific Plot

```python
get_plot(run_id="a1b2c3d4e5f6g7h8", name="residuals")
```

Returns `{"path": "/absolute/path/to/residuals.png", "name": "residuals", "description": "..."}`.

## Step 5: Compare Models

After fitting multiple models, compare them:

```python
compare_models(run_ids=["run_id_1", "run_id_2", "run_id_3"])
```

Returns a `DiagnosticResult` dict with the comparison table in `detail.models` and the best model identified in `interpretation`.

## Data Flow Summary

```
register_dataset  -->  [DatasetCatalog]  -->  query_data / fit_model
                                                    |
                                              [Engine: fit]
                                                    |
                                         [Diagnostics + Plots]
                                                    |
                                         [ArtifactStore + MetaStore]
                                                    |
                                      get_run / list_artifacts / get_plot
```

## Key Conventions

1. **Dataset names** are strings you choose. Use dots for namespacing: `"equities.daily"`, `"macro.gdp"`.
2. **Run IDs** are auto-generated hex strings (16 chars). Always returned by `fit_model`.
3. **All tools return dicts.** Never raises. Check for `"error": true` in the return value.
4. **Artifacts are stored on disk** under `{artifacts_dir}/{run_id}/plots/`, `{artifacts_dir}/{run_id}/data/`, `{artifacts_dir}/{run_id}/summaries/`.
5. **Plots are auto-generated on fit.** You do not need to call a separate plot tool. Use `get_plot` to retrieve paths after fitting.
6. **Forecasts can be generated at fit time** (via `forecast_steps`) or after the fact (via `forecast_model` with a `run_id`).

## Typical Agent Workflow

1. Call `register_dataset` for each data file.
2. Call `list_datasets` or `profile_dataset` to understand the data.
3. Call `run_diagnostics` to check stationarity if fitting time-series models.
4. Call `fit_model` with appropriate family and params.
5. Read `run_id`, `status`, `metrics_json`, and `diagnostics_json` from the return.
6. Call `get_plot` to retrieve visualization paths.
7. Optionally call `forecast_model` for additional forecasts.
8. Call `compare_models` if multiple models were fitted.
