# Data Downloader Overview

## What It Does

The data_downloader module provides access to a curated 140-ticker financial universe spanning 7 asset classes, with automatic data fetching from Yahoo Finance, FRED, and ECB. Data is cached as parquet files and auto-registered in the stat_runtime for immediate analysis.

## Three LLM Tools

| Tool | Purpose | When to use |
|---|---|---|
| `list_universe` | Browse tickers by asset class | First step — see what's available |
| `search_tickers` | Search by name, symbol, sector, country | Find specific tickers |
| `download_tickers` | Download data and register in stat_runtime | Before any analysis |

## 140-Ticker Universe — 7 Asset Classes

| Asset Class | Count | Examples | Source |
|---|---|---|---|
| EQUITY | 47 | SPY, QQQ, AAPL, IEMG, VEA, FXI | Yahoo Finance |
| FIXED_INCOME | 21 | AGG, TLT, IEF, LQD, HYG, EMB | Yahoo Finance |
| COMMODITIES | 17 | GLD, SLV, USO, UNG, DBC, CORN | Yahoo Finance + FRED |
| MACRO | 34 | DGS10, CPIAUCSL, UNRATE, GDP, EFFR | FRED |
| FX | 10 | EURUSD=X, GBPUSD=X, USDJPY=X, DTWEXBGS | Yahoo + FRED |
| ALTERNATIVES | 9 | BTC-USD, ETH-USD, VIXCLS, IBIT | Yahoo + FRED |
| REAL_ESTATE | 2 | VNQ, VNQI | Yahoo Finance |

## Data Sources

| Source | Tickers | API Key | Method |
|---|---|---|---|
| Yahoo Finance | ~95 tickers | Not needed | yfinance batch download |
| FRED | ~38 tickers | Not needed | Direct CSV endpoint |
| ECB | ~5 tickers | Not needed | ECB SDW CSV API |

Source is auto-detected from the ticker universe database — the LLM does not need to specify it.

## 4-Layer Taxonomy

Each ticker is classified in 4 layers:

```
Layer1: Asset Class      → EQUITY, FIXED_INCOME, COMMODITIES, MACRO, FX, ALTERNATIVES, REAL_ESTATE
Layer2: Sub-Asset Class  → Developed, Emerging, Government, Corporate, Energy, Metals, Crypto
Layer3: Geographic/Sector → US, Europe, Japan, China, Technology, Healthcare, Oil, Gold
Layer4: Granular          → Large-Cap, Small-Cap, Growth, Value, 1-3Y, 20+Y, WTI-Crude
```

## Default Workflow

```
1. search_tickers("US equity large cap")     → find relevant tickers
2. download_tickers(["SPY", "QQQ", "IWM"])   → download + auto-register
3. discover_data()                            → verify datasets, see column roles
4. analyze(dataset_name="spy", mode="volatility")  → run analysis
```
