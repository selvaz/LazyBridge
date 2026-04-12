# Persistence Contract & Tool Architecture

This document describes what each tool persists, the artifact directory layout, the tool dependency graph, and the recommended LLM workflow. It is the operational reference for understanding side effects and tool chaining.

---

## Two-Tier Tool Architecture

Tools are split into two tiers:

### High-Level Tools (Main Agent)

| Tool | Purpose |
|---|---|
| `discover_data()` | See all registered datasets with column roles, types, and suggestions |
| `discover_analyses()` | See all completed runs with inline metrics, diagnostics, and artifact catalogs |
| `analyze()` | Run a complete analysis: fit + diagnostics + plots + interpretation |
| `register_dataset()` | Register a new data file for analysis |

### Low-Level Expert Tools (Sub-Agent / Expert Mode)

| Tool | Purpose |
|---|---|
| `fit_model` | Fit a specific model with custom parameters |
| `forecast_model` | Generate a forecast from a past run |
| `query_data` | Execute SQL against registered datasets |
| `profile_dataset` | Compute column-level statistics |
| `run_diagnostics` | Run ADF + KPSS stationarity tests |
| `compare_models` | Compare multiple runs by AIC/BIC |
| `get_run` | Retrieve full run record |
| `list_runs` | List past runs |
| `list_datasets` | List registered datasets |
| `list_artifacts` | List artifacts for a run |
| `get_plot` | Get a specific plot path |

When using `stat_agent(expert_mode=True)`, the main agent delegates to the expert sub-agent via `delegate_to_expert(instruction)`.

---

## Tool Side-Effect Matrix

| Tool | MetaStore Writes | ArtifactStore Writes | Notes |
|---|---|---|---|
| `register_dataset` | `save_dataset(DatasetMeta)` | None | Scans file schema + row count via Polars lazy scan. Does NOT load data. |
| `profile_dataset` | `save_dataset(meta)` — updates `profile_json` | None | Loads FULL dataset into memory. Caches profile in metadata. |
| `fit_model` / `analyze` | `save_run()` x2 (RUNNING → SUCCESS/FAILED), `save_artifact()` x N | `plots/*.png`, `data/*.json`, `summaries/*.json` | N = 5-10 artifacts depending on family + forecast_steps |
| `forecast_model` | None | None | Re-fits model from stored spec_json. Returns data only. |
| `query_data` | None | None | Pure read. SQL validated + executed via DuckDB. |
| `run_diagnostics` | None | None | Pure compute. ADF + KPSS tests on a column. |
| `compare_models` | None | None | Pure read. Compares metrics across existing runs. |
| `get_run` | None | None | Pure read. |
| `list_runs` | None | None | Pure read. |
| `list_datasets` | None | None | Pure read. |
| `list_artifacts` | None | None | Pure read. |
| `get_plot` | None | None | Pure read. Returns file path. |
| `discover_data` | None | None | Pure read. Enriches with column role inference. |
| `discover_analyses` | None | None | Pure read. Enriches runs with inline artifacts. |

**Key insight**: Only `register_dataset`, `profile_dataset`, `fit_model`, and `analyze` have side effects. All other tools are pure reads.

---

## Artifact Directory Layout

When `fit_model` or `analyze` runs successfully, artifacts are stored on disk:

```
{artifacts_dir}/
  {run_id}/
    plots/
      residuals.png         # Always generated
      acf_pacf.png          # Always generated
      volatility.png        # GARCH family only
      regimes.png           # Markov family only
      forecast.png          # Only if forecast_steps > 0
    data/
      residuals.json        # Residual values as JSON array
      forecast.json         # Only if forecast_steps > 0
    summaries/
      spec.json             # Original ModelSpec
      fit_summary.txt       # Model summary text
      diagnostics.json      # All diagnostic results
```

Each artifact is also registered in MetaStore via `save_artifact(ArtifactRecord)`.

---

## MetaStore Persistence

Two backends with identical API:

| Backend | When | Data Lifetime |
|---|---|---|
| **InMemory** | `StatRuntime()` (default) | Process lifetime only |
| **DuckDB** | `StatRuntime(db="path.duckdb")` | Persistent on disk |

### Tables (DuckDB backend)

- `datasets` — registered datasets (name, uri, schema, frequency, time_column, profile_json)
- `runs` — model run records (run_id, spec, status, metrics, diagnostics, artifact_paths)
- `artifacts` — artifact metadata (run_id, name, type, path, description)

---

## Tool Dependency Graph

```
register_dataset ──> [Required before ANY data tool]
    ├── discover_data        (works with 0 datasets but trivial)
    ├── list_datasets
    ├── profile_dataset
    ├── query_data
    ├── run_diagnostics
    ├── fit_model / analyze ──> [Creates run_id, required for]
    │       ├── get_run
    │       ├── list_runs
    │       ├── list_artifacts
    │       ├── get_plot
    │       ├── forecast_model (requires same-session model object)
    │       └── compare_models (needs 2+ run_ids)
    └── discover_analyses    (works with 0 runs but trivial)
```

**Critical**: `forecast_model` re-fits the model from stored `spec_json`. The fitted model object is NOT persisted — it's held in memory only during the `execute()` call.

---

## Recommended LLM Workflow

### Standard Flow (high-level tools)

```
1. discover_data()           → "What data exists?"
2. discover_analyses()       → "What's already been done?"
3. analyze(...)              → Run new analysis
4. discover_analyses()       → See updated results
5. delegate_to_expert(...)   → Fine-grained control if needed
```

### Expert Flow (low-level tools)

```
1. register_dataset(...)     → Make data available
2. profile_dataset(...)      → Inspect column statistics
3. run_diagnostics(...)      → Check stationarity
4. fit_model(...)            → Fit model with custom params
5. list_artifacts(...)       → See what was generated
6. compare_models(...)       → Pick best model
7. forecast_model(...)       → Extended forecast
```

### When to Delegate to Expert

- User wants specific ARIMA(2,1,3) parameters
- User wants custom SQL to join/filter data
- User wants to compare specific run IDs manually
- User wants a specific plot regenerated
- User wants to run stationarity tests independently
