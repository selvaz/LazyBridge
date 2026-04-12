# Data Downloader Module

## Overview

The `data_downloader` module provides a 140-ticker financial data universe with automatic fetching from Yahoo Finance, FRED, and ECB. It integrates directly with the `stat_runtime` for seamless data registration and analysis.

## Installation

```bash
pip install lazybridge[stats]
pip install yfinance requests pyarrow
```

## Quick Start

### Python API

```python
from lazybridge.data_downloader import TickerDatabase, DataDownloader

# Browse the universe
db = TickerDatabase()
print(db.summary())                          # 140 tickers, 7 asset classes
print(db.search("treasury"))                 # Find treasury tickers
print(db.filter(asset_class="EQUITY"))       # All equity tickers

# Download data
dl = DataDownloader(cache_dir="./ticker_cache")
result = dl.fetch("SPY", "YAHOO", "2020-01-01")
print(result.rows, result.file_path)
```

### With LLM Agent

```python
from lazybridge.stat_runtime.tools import stat_agent

agent, rt = stat_agent("anthropic", include_downloader=True)
resp = agent.loop("Download SPY and AAPL, analyze their volatility")
print(resp.content)
rt.close()
```

### With stat_runtime (manual)

```python
from lazybridge.stat_runtime.runner import StatRuntime
from lazybridge.stat_runtime.tools import stat_tools
from lazybridge.data_downloader import TickerDatabase, DataDownloader, downloader_tools

rt = StatRuntime(artifacts_dir="./artifacts")
db = TickerDatabase()
dl = DataDownloader(cache_dir="./ticker_cache")

# Combine tools
all_tools = downloader_tools(rt, db, dl) + stat_tools(rt, level="high")
```

## Ticker Universe

140 curated tickers across 7 asset classes:

| Asset Class | Count | Source | Examples |
|---|---|---|---|
| EQUITY | 47 | Yahoo Finance | SPY, QQQ, IWM, IEMG, VEA, FXI, XLK |
| FIXED_INCOME | 21 | Yahoo Finance | AGG, TLT, IEF, LQD, HYG, EMB |
| MACRO | 34 | FRED | DGS10, CPIAUCSL, UNRATE, GDP, EFFR |
| COMMODITIES | 17 | Yahoo + FRED | GLD, SLV, USO, UNG, DBC, DCOILWTICO |
| FX | 10 | Yahoo + FRED | EURUSD=X, GBPUSD=X, DTWEXBGS |
| ALTERNATIVES | 9 | Yahoo + FRED | BTC-USD, ETH-USD, VIXCLS, IBIT |
| REAL_ESTATE | 2 | Yahoo Finance | VNQ, VNQI |

### 4-Layer Taxonomy

```
Layer 1: Asset Class        → EQUITY, FIXED_INCOME, COMMODITIES, MACRO, FX, ...
Layer 2: Sub-Asset Class    → Developed, Emerging, Government, Corporate, Energy, ...
Layer 3: Geographic/Sector  → US, Europe, Japan, China, Technology, Healthcare, ...
Layer 4: Granular           → Large-Cap, Small-Cap, Growth, Value, 1-3Y, 20+Y, ...
```

## Data Sources

### Yahoo Finance
- 95+ tickers: equities, ETFs, commodities, crypto, FX pairs
- No API key required
- Batch download for efficiency
- Adjusted close prices

### FRED (Federal Reserve Economic Data)
- 38 tickers: interest rates, inflation, GDP, unemployment, money supply
- No API key required
- Direct CSV download from fred.stlouisfed.org

### ECB (European Central Bank)
- 5 tickers: ECB policy rates, EU GDP
- No API key required
- ECB Statistical Data Warehouse API

## LLM Tools

### list_universe(asset_class=None, sub_asset_class=None)
Browse available tickers grouped by asset class.

### search_tickers(query)
Search by name, symbol, sector, country. Partial matches supported.

### download_tickers(tickers, start=None, end=None, auto_register=True)
Download and auto-register in stat_runtime for analysis.

## Data Format

All downloaded data is standardized to two columns:

| Column | Type | Description |
|---|---|---|
| date | Date | ISO date (YYYY-MM-DD) |
| value | Float64 | Close price or indicator value |

Stored as parquet files in `{cache_dir}/{TICKER}.parquet`.

## Caching

- Per-ticker parquet files with incremental updates
- Re-running download only fetches new data since last fetch
- Merge/dedup logic handles revisions (keeps latest value per date)
- Cache directory defaults to `{artifacts_dir}/ticker_cache/`

## API Reference

### TickerDatabase

```python
db = TickerDatabase()
db.list_all()                                    # list[TickerInfo] — all 140
db.get("SPY")                                    # TickerInfo | None
db.search("gold")                                # list[TickerInfo]
db.filter(asset_class="EQUITY", source="YAHOO")  # list[TickerInfo]
db.by_asset_class()                              # dict[str, list[TickerInfo]]
db.summary()                                     # compact dict for display
```

### DataDownloader

```python
dl = DataDownloader(cache_dir="./cache")
dl.fetch("SPY", "YAHOO", "2020-01-01")           # FetchResult
dl.fetch_batch(ticker_infos, start, end)          # list[FetchResult]
dl.download_and_register(ticker_infos, start, end, runtime)  # list[FetchResult]
```

### Schemas

```python
from lazybridge.data_downloader.schemas import (
    TickerInfo,         # Ticker metadata (symbol, name, source, taxonomy)
    FetchResult,        # Download result (ok, rows, path, frequency)
    DownloaderConfig,   # Config (timeout, retries, cache settings)
)
```
