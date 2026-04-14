# Discovery Tools Reference

## discover_data()

### What It Does

Returns enriched metadata for all registered datasets. This is the recommended first tool call in any analysis workflow. It provides column types, inferred semantic roles (target, time, entity key, etc.), quality signals from cached profiles, a natural language summary, and actionable suggestions.

### Tool Signature

```python
discover_data() -> dict
```

No parameters. Returns a `DataDiscoveryResult` dict.

### Return Value

```json
{
  "datasets": [
    {
      "name": "equities",
      "uri": "/data/returns.parquet",
      "file_format": "parquet",
      "frequency": "daily",
      "row_count": 5000,
      "time_column": "date",
      "entity_keys": ["symbol"],
      "columns": {"date": "Date", "symbol": "Utf8", "ret": "Float64", "volume": "Int64"},
      "column_roles": [
        {"column": "date", "dtype": "Date", "inferred_role": "time", "confidence": "high", "reason": "Declared as time_column"},
        {"column": "symbol", "dtype": "Utf8", "inferred_role": "entity_key", "confidence": "high", "reason": "Declared in entity_keys"},
        {"column": "ret", "dtype": "Float64", "inferred_role": "target", "confidence": "medium", "reason": "Numeric column named 'ret' matches return patterns"},
        {"column": "volume", "dtype": "Int64", "inferred_role": "exogenous", "confidence": "low", "reason": "Numeric column not matching known target patterns"}
      ],
      "column_signals": {
        "ret": {"null_pct": 0.01, "unique_count": 4500, "mean": 0.0004, "min_val": null, "max_val": null},
        "volume": {"null_pct": 0.0, "unique_count": 3000, "mean": null, "min_val": null, "max_val": null}
      },
      "suggestions": [
        "Likely target column: 'ret' (Numeric column named 'ret' matches common target variable patterns).",
        "Panel data detected (entity_keys: [symbol]). Filter by entity using query_sql before fitting."
      ],
      "has_profile": true,
      "business_description": null,
      "canonical_target": null,
      "summary": "equities: 5,000 daily observations; time column: date; likely target(s): ret; grouped by symbol; 2 numeric columns."
    }
  ],
  "total_datasets": 1,
  "suggestions": []
}
```

### Column Roles

The inference engine assigns one of these roles to each column:

| Role | Meaning | How detected |
|---|---|---|
| `time` | Time/date column | Datetime dtype, or name matches (date, time, timestamp, etc.), or declared `time_column` |
| `target` | Likely dependent variable | Float64 + name matches (ret, return, close, price, value, yield, growth, etc.) |
| `entity_key` | Panel data key | Declared in `entity_keys`, or string column named symbol/ticker/country/sector |
| `exogenous` | Numeric feature | Numeric column not classified as target |
| `identifier` | ID column (ignore in modeling) | String column named id/code/isin/cusip |
| `unknown` | Unclassified | Everything else |

### Column Signals (from cached profile)

If `profile_dataset()` has been called, signals are populated:

| Signal | Meaning |
|---|---|
| `null_pct` | Fraction of null values (0.0 to 1.0) |
| `unique_count` | Number of distinct values |
| `min_val` | Minimum value (numeric columns) |
| `max_val` | Maximum value (numeric columns) |
| `mean` | Mean value (numeric columns) |

If no profile is cached, `column_signals` is empty and `has_profile` is `false`.

---

## discover_analyses()

### What It Does

Returns all completed analysis runs with inline metrics, diagnostics summaries, and a full artifact catalog per run. Use this to see what has already been done before running new analyses.

### Tool Signature

```python
discover_analyses(
    dataset_name: str | None = None,  # Filter to one dataset
    limit: int = 20,                  # Max runs to return
) -> dict
```

### Return Value

```json
{
  "runs": [
    {
      "run_id": "a1b2c3d4e5f6g7h8",
      "dataset_name": "equities",
      "engine": "garch",
      "status": "success",
      "created_at": "2025-01-15T10:31:00+00:00",
      "duration_secs": 2.5,
      "aic": 1234.5,
      "bic": 1267.8,
      "log_likelihood": -612.25,
      "diagnostics_passed": 2,
      "diagnostics_failed": 0,
      "diagnostics_total": 2,
      "target_col": "ret",
      "model_params": {"p": 1, "q": 1},
      "artifacts": [
        {"name": "residuals", "artifact_type": "plot", "file_format": "png", "path": "artifacts/.../plots/residuals.png", "description": "Residual analysis"},
        {"name": "volatility", "artifact_type": "plot", "file_format": "png", "path": "artifacts/.../plots/volatility.png", "description": "Conditional volatility"}
      ],
      "error_message": null
    }
  ],
  "total_runs": 1,
  "datasets_analyzed": ["equities"],
  "best_by_aic": "a1b2c3d4e5f6g7h8",
  "best_by_bic": "a1b2c3d4e5f6g7h8",
  "suggestions": []
}
```

### Key Fields

- `best_by_aic` / `best_by_bic`: run_id of the best model across all returned runs.
- `artifacts`: Complete list of plots, data files, and summaries for each run — no need to call `list_artifacts` separately.
- `suggestions`: Actionable hints (e.g., "3 runs failed", "Multiple families fitted — use compare_models()").
