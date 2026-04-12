# Data Downloader Tools Reference

## list_universe()

### Signature

```python
list_universe(
    asset_class: str | None = None,   # EQUITY, FIXED_INCOME, COMMODITIES, MACRO, FX, ALTERNATIVES, REAL_ESTATE
    sub_asset_class: str | None = None,  # Developed, Emerging, Government, Corporate, Energy, Metals, Crypto
)
```

### When to Use

Call first to browse available tickers. Without filters, returns a grouped summary with counts per asset class. With filters, returns detailed ticker lists.

### Examples

```python
# See everything
list_universe()
# → {total_tickers: 140, asset_classes: {EQUITY: {count: 47, tickers: [...]}, ...}}

# Filter to equities
list_universe(asset_class="EQUITY")
# → {total: 47, tickers: [{ticker: "SPY", name: "S&P 500", ...}, ...]}

# Filter to emerging market equities
list_universe(asset_class="EQUITY", sub_asset_class="Emerging")
# → {total: 8, tickers: [{ticker: "IEMG"}, {ticker: "FXI"}, ...]}
```

### Return Value

Without filters:
```json
{
  "total_tickers": 140,
  "asset_classes": {
    "EQUITY": {"count": 47, "tickers": ["SPY", "QQQ", ...]},
    "MACRO": {"count": 34, "tickers": ["DGS10", "CPIAUCSL", ...]},
    ...
  },
  "sources": {"YAHOO": 95, "FRED": 38, "ECB": 5, "UNKNOWN": 2}
}
```

With filters:
```json
{
  "total": 47,
  "filters_applied": {"asset_class": "EQUITY"},
  "tickers": [
    {"ticker": "SPY", "name": "EQUITY | US | S&P 500", "source": "YAHOO",
     "asset_class": "EQUITY", "sub_asset": "Developed", "geo_sector": "US", "granular": "Large-Cap"}
  ]
}
```

---

## search_tickers()

### Signature

```python
search_tickers(
    query: str,  # Search across ticker symbol, name, asset class, sector, country
)
```

### When to Use

When looking for specific tickers by keyword. Searches across all fields (ticker, name, asset class, sub-asset, geographic sector, granular, area).

### Examples

```python
search_tickers("gold")       # → GLD (SPDR Gold Trust)
search_tickers("treasury")   # → SHY, IEI, IEF, TLT, DGS10, DGS2, ...
search_tickers("emerging")   # → IEMG, FXI, INDA, EWZ, EMB, EMLC, ...
search_tickers("technology")  # → XLK (Technology Select Sector SPDR)
search_tickers("bitcoin")    # → BTC-USD, IBIT
search_tickers("SPY")        # → SPY (S&P 500)
search_tickers("inflation")  # → CPIAUCSL, CPILFESL, T5YIE, T10YIE
```

### Return Value

```json
{
  "query": "gold",
  "matches": 1,
  "tickers": [
    {"ticker": "GLD", "name": "CMDTY | Metals | SPDR Gold Trust", "source": "YAHOO",
     "asset_class": "COMMODITIES", "sub_asset": "Metals (Precious)", "geo_sector": "Gold"}
  ]
}
```

---

## download_tickers()

### Signature

```python
download_tickers(
    tickers: list[str],         # Ticker symbols (e.g. ["SPY", "AAPL", "DGS10"])
    start: str | None = None,   # Start date YYYY-MM-DD (default: 2000-01-01)
    end: str | None = None,     # End date YYYY-MM-DD (default: today)
    auto_register: bool = True, # Auto-register in stat_runtime for analysis
)
```

### When to Use

After finding tickers via search_tickers or list_universe. Downloads data from the appropriate source (auto-detected) and registers each ticker as a dataset in stat_runtime.

### Key Behaviors

- **Auto source detection**: Looks up each ticker in the universe database to determine Yahoo/FRED/ECB
- **Unknown tickers**: If not in universe, defaults to Yahoo Finance
- **Parquet caching**: Data is saved as per-ticker parquet files for incremental updates
- **Incremental updates**: Only downloads new data since last fetch
- **Auto-registration**: When `auto_register=True`, each ticker becomes a stat_runtime dataset

### Examples

```python
# Download and register US equities
download_tickers(["SPY", "QQQ", "IWM"], start="2020-01-01")
# → {succeeded: 3, registered_datasets: ["spy", "qqq", "iwm"]}

# Download macro data
download_tickers(["DGS10", "CPIAUCSL", "UNRATE"])
# → {succeeded: 3, registered_datasets: ["dgs10", "cpiaucsl", "unrate"]}

# Download without registering
download_tickers(["SPY"], auto_register=False)
# → {succeeded: 1, registered_datasets: []}
```

### Return Value

```json
{
  "total": 3,
  "succeeded": 3,
  "failed": 0,
  "results": [
    {"ticker": "SPY", "source": "YAHOO", "ok": true, "rows": 6300,
     "date_start": "2000-01-03", "date_end": "2024-04-12",
     "file_path": "/path/to/ticker_cache/SPY.parquet",
     "frequency": "daily", "registered_as": "spy"}
  ],
  "registered_datasets": ["spy", "qqq", "iwm"]
}
```

### After Downloading

The registered datasets are immediately available for:
- `discover_data()` — see column roles, quality signals
- `analyze(dataset_name="spy", mode="volatility")` — run analysis
- `query_data(sql="SELECT * FROM dataset('spy') WHERE ...")` — SQL queries

### Dataset Registration Details

Each downloaded ticker is registered with:
- `name`: lowercase ticker symbol (e.g., "spy", "dgs10")
- `time_column`: "date" (always)
- `frequency`: auto-detected (daily, weekly, monthly, quarterly)
- `business_description`: from ticker universe name (e.g., "EQUITY | US | S&P 500")
- `columns`: "date" (Date), "value" (Float64) — close price or indicator value

### Data Format

All downloaded data has two columns:

| Column | Type | Description |
|---|---|---|
| date | Date | ISO date |
| value | Float64 | Close/adjusted price (equities) or indicator value (macro) |

For analysis, use `target_col="value"` in analyze().
