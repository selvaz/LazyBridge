# query_data Tool Reference

## Purpose
Execute SQL queries against registered datasets using DuckDB.

## Syntax
```
query_data(sql, max_rows=5000)
```

## Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| sql | str | required | SQL SELECT query with dataset() macros |
| max_rows | int | 5000 | Maximum rows to return |

## The dataset() Macro
Reference registered datasets by name:
```sql
SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY' ORDER BY date
```

The macro expands to `read_parquet('/path/to/file.parquet')` internally.

## Rules
1. Only SELECT and WITH (CTE) statements allowed
2. INSERT, UPDATE, DELETE, DROP, ALTER, CREATE are blocked
3. ORDER BY is recommended for time-series data (warning if missing)
4. Results are truncated at max_rows with a `truncated` flag

## Return Format
```json
{
  "query_hash": "abc123def456",
  "original_sql": "SELECT ...",
  "normalized_sql": "SELECT ...",
  "columns": ["date", "ret"],
  "row_count": 1250,
  "truncated": false,
  "data_json": [{"date": "2024-01-02", "ret": 0.0123}, ...]
}
```

## Examples

### Basic query
```
query_data("SELECT date, ret FROM dataset('equities') ORDER BY date")
```

### Filtered query
```
query_data("SELECT * FROM dataset('equities') WHERE symbol IN ('SPY', 'QQQ') AND date >= '2023-01-01'")
```

### Aggregation
```
query_data("SELECT symbol, AVG(ret) as avg_ret, STDDEV(ret) as vol FROM dataset('equities') GROUP BY symbol")
```

### CTE query
```sql
WITH daily AS (
  SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY'
)
SELECT date, ret, AVG(ret) OVER (ORDER BY date ROWS 20 PRECEDING) as ma20
FROM daily ORDER BY date
```

## Common Errors
- `Dataset 'X' is not registered` — register the dataset first
- `Forbidden SQL keyword detected: INSERT` — only SELECT allowed
- `Only SELECT statements are allowed` — query must start with SELECT or WITH
