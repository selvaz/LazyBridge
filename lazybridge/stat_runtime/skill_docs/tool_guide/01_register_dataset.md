# register_dataset Tool Reference

## What It Does

`register_dataset` scans a Parquet or CSV file on disk, extracts the column schema and row count via Polars lazy scan, and stores the resulting `DatasetMeta` object in the runtime catalog. The file is NOT loaded into memory -- only metadata is read. After registration, the dataset can be referenced by name in `fit_model`, `query_data`, `profile_dataset`, and other tools.

If a dataset with the same name already exists, it is silently overwritten (the old entry is deleted and replaced).

## Tool Signature

```python
register_dataset(
    name: str,                              # Required
    uri: str,                               # Required
    time_column: str | None = None,         # Optional
    frequency: str = "daily",               # Optional
    entity_keys: list[str] | None = None,   # Optional
)
```

## Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | `str` | Yes | -- | Logical name for the dataset. Used to reference it in all subsequent tool calls. Use dot-delimited namespacing for clarity (e.g. `"equities.daily"`, `"macro.gdp"`). Any string is valid. |
| `uri` | `str` | Yes | -- | Absolute or relative file path to a Parquet or CSV file. The file extension determines the format: `.csv` triggers CSV registration, everything else triggers Parquet registration. The path is resolved to an absolute path internally via `Path(uri).resolve()`. |
| `time_column` | `str \| None` | No | `None` | Name of the primary time/date column. If provided, the column must exist in the file schema or a `ValueError` is raised. When set, `fit_model` uses this column to sort data in temporal order before fitting time-series models. If not set and a `fit_model` call does not provide `time_col`, data is used in file order. |
| `frequency` | `str` | No | `"daily"` | Data frequency. Must be one of: `"daily"`, `"weekly"`, `"monthly"`, `"quarterly"`, `"annual"`, `"intraday"`, `"irregular"`. Stored as metadata; does not affect computation. |
| `entity_keys` | `list[str] \| None` | No | `None` | Panel data key columns (e.g. `["symbol"]`, `["country", "sector"]`). Stored as metadata. Useful for documenting multi-entity datasets. Does not affect computation directly. |

## Supported Formats

### Parquet

- Scanned via `polars.scan_parquet()` (lazy -- reads only file metadata).
- Schema is extracted from the Parquet footer without loading data.
- Row count is computed via `lf.select(pl.len()).collect().item()`.
- Supports single files and Hive-partitioned directories.

### CSV

- Scanned via `polars.scan_csv()` (lazy scan).
- Schema is inferred from the CSV by Polars (type inference on a sample).
- Row count is computed the same way as Parquet.
- CSV files must have a header row.

## What Gets Stored

A `DatasetMeta` Pydantic model is persisted in the `MetaStore` (in-memory dict or DuckDB `datasets` table, depending on runtime configuration).

### DatasetMeta Fields

| Field | Type | Source | Description |
|---|---|---|---|
| `dataset_id` | `str` | Auto-generated | 12-character hex UUID, e.g. `"a1b2c3d4e5f6"` |
| `name` | `str` | User-provided | The logical name passed to `register_dataset` |
| `version` | `str` | Default `"1"` | Always `"1"` when registered via the tool |
| `uri` | `str` | User-provided (resolved) | Absolute file path after `Path.resolve()` |
| `file_format` | `str` | Auto-detected | `"parquet"` or `"csv"` based on file extension |
| `columns_schema` | `dict[str, str]` | Auto-detected | Column name to Polars dtype string mapping (e.g. `{"date": "Date", "ret": "Float64", "symbol": "Utf8"}`) |
| `frequency` | `Frequency` | User-provided | The frequency enum value |
| `time_column` | `str \| None` | User-provided | The time column name, or `None` |
| `entity_keys` | `list[str]` | User-provided | Entity key column names, or empty list |
| `semantic_roles` | `dict[str, str]` | Default `{}` | Not populated by the tool; reserved for future use |
| `profile_json` | `dict` | Default `{}` | Empty until `profile_dataset` is called |
| `row_count` | `int` | Auto-detected | Total number of rows in the file |
| `registered_at` | `datetime` | Auto-generated | UTC timestamp of registration |

## Return Value

On success, returns the full `DatasetMeta` serialized as a JSON dict via `model_dump(mode="json")`:

```json
{
  "dataset_id": "a1b2c3d4e5f6",
  "name": "equities",
  "version": "1",
  "uri": "/absolute/path/to/returns.parquet",
  "file_format": "parquet",
  "columns_schema": {
    "date": "Date",
    "symbol": "Utf8",
    "ret": "Float64",
    "volume": "Int64"
  },
  "frequency": "daily",
  "time_column": "date",
  "entity_keys": ["symbol"],
  "semantic_roles": {},
  "profile_json": {},
  "row_count": 125000,
  "registered_at": "2025-01-15T10:30:00+00:00"
}
```

On error, returns an error dict. The tool never raises exceptions.

```json
{"error": true, "type": "FileNotFoundError", "message": "Parquet path does not exist: /data/returns.parquet"}
```

## Auto-Detection Details

### Schema Detection (columns_schema)

The schema is extracted from the Polars lazy frame's `collect_schema()` method. This reads the Parquet file footer or infers types from the CSV. The result maps each column name to its Polars dtype as a string.

Common Polars dtype strings you will see:
- `"Float64"`, `"Float32"` -- floating point numbers
- `"Int64"`, `"Int32"`, `"Int16"`, `"Int8"` -- integers
- `"Utf8"` or `"String"` -- string/text columns
- `"Date"` -- date without time
- `"Datetime(time_unit='us', time_zone=None)"` -- datetime
- `"Boolean"` -- true/false

### Row Count Detection (row_count)

Computed via `lf.select(pl.len()).collect().item()`. For Parquet files, this reads the row group metadata and is very fast even for large files. For CSV files, the entire file must be scanned.

## Common Errors

| Error Type | Cause | Message Pattern |
|---|---|---|
| `FileNotFoundError` | File does not exist at the given path | `"Parquet path does not exist: /path/..."` or `"CSV path does not exist: /path/..."` |
| `ValueError` | `time_column` name not found in file schema | `"time_column 'X' not found in schema. Available columns: [...]"` |
| `ImportError` | Polars is not installed | `"polars is required for this feature. Run: pip install lazybridge[stats]"` |
| `polars.exceptions.ComputeError` | File is corrupt or not valid Parquet/CSV | Varies by error |

### Duplicate Name Behavior

Registering a dataset with the same name as an existing entry silently overwrites it. The `MetaStore.save_dataset` method deletes the old entry by name before inserting the new one. The old `dataset_id` is discarded and a new one is generated. There is no warning or error.

## Examples

### Register a Parquet File

```python
register_dataset(
    name="equities",
    uri="/data/returns.parquet",
    time_column="date",
    frequency="daily",
    entity_keys=["symbol"]
)
```

### Register a CSV File

```python
register_dataset(
    name="macro.gdp",
    uri="/data/gdp_quarterly.csv",
    time_column="quarter",
    frequency="quarterly"
)
```

### Register Without Time Column

```python
register_dataset(
    name="cross_section",
    uri="/data/firms_2024.parquet",
    frequency="irregular"
)
```

This is valid for cross-sectional data or when temporal ordering is not needed.

### Register With Multiple Entity Keys

```python
register_dataset(
    name="panel.equities",
    uri="/data/international_returns.parquet",
    time_column="date",
    frequency="daily",
    entity_keys=["country", "symbol"]
)
```

### Overwrite an Existing Registration

```python
# First registration
register_dataset(name="prices", uri="/data/prices_v1.parquet")

# This overwrites the first one -- no error
register_dataset(name="prices", uri="/data/prices_v2.parquet", time_column="date")
```

## What Happens After Registration

After a successful `register_dataset` call, the dataset name can be used in:

- `list_datasets()` -- will appear in the listing
- `profile_dataset(name="...")` -- computes column-level statistics
- `query_data(sql="SELECT ... FROM dataset('name') ...")` -- SQL access
- `fit_model(dataset_name="...")` -- model fitting
- `run_diagnostics(series_name="...")` -- stationarity tests

The file is NOT loaded into memory at registration time. Data loading happens on demand when one of the above tools is called.

## Persistence Behavior

- **In-memory runtime** (`StatRuntime()`): The `DatasetMeta` is stored in a Python dict keyed by name. Lost when the process exits.
- **DuckDB runtime** (`StatRuntime(db="file.duckdb")`): The `DatasetMeta` is inserted into the `datasets` table. Survives process restarts. The actual data file must still exist at the `uri` path for subsequent operations.
