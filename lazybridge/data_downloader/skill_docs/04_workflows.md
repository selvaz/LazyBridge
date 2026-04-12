# Data Downloader Workflows

## Workflow A: Equity Volatility Analysis

**Goal**: Download US equity ETFs and analyze volatility.

```
Step 1: search_tickers("US equity large cap")
  → SPY, QQQ, IWM, IWF, IWD

Step 2: download_tickers(["SPY", "QQQ", "IWM"], start="2020-01-01")
  → Downloads 3 tickers, registers as "spy", "qqq", "iwm"

Step 3: discover_data()
  → Shows 3 datasets, each with date (time) and value (target) columns

Step 4: analyze(dataset_name="spy", target_col="value", mode="volatility")
  → GARCH analysis on SPY prices
  → Check interpretation, model_adequate, volatility persistence

Step 5: analyze(dataset_name="qqq", target_col="value", mode="volatility")
  → Compare with QQQ

Step 6: discover_analyses()
  → Review all runs, compare AIC/BIC across tickers
```

## Workflow B: Macro Regime Detection

**Goal**: Detect regimes in treasury yields.

```
Step 1: search_tickers("treasury rate")
  → DGS10, DGS2, DGS30, DGS5, DGS3MO

Step 2: download_tickers(["DGS10", "DGS2"], start="2000-01-01")
  → Downloads from FRED, registers as "dgs10", "dgs2"

Step 3: analyze(dataset_name="dgs10", target_col="value", mode="regime")
  → Markov switching model on 10-year treasury rates
  → See regime probabilities, transition matrix, expected durations

Step 4: analyze(dataset_name="dgs2", target_col="value", mode="regime")
  → Compare short-term rate regimes
```

## Workflow C: Cross-Asset Comparison

**Goal**: Compare volatility across asset classes.

```
Step 1: list_universe()
  → See all 7 asset classes with counts

Step 2: download_tickers([
    "SPY",        # US equity
    "AGG",        # US bonds
    "GLD",        # Gold
    "BTC-USD",    # Bitcoin
], start="2020-01-01")

Step 3: For each dataset:
    analyze(dataset_name="spy", target_col="value", mode="volatility")
    analyze(dataset_name="agg", target_col="value", mode="volatility")
    analyze(dataset_name="gld", target_col="value", mode="volatility")
    analyze(dataset_name="btc-usd", target_col="value", mode="volatility")

Step 4: discover_analyses()
  → Compare volatility persistence across asset classes
```

## Workflow D: Forecasting Macro Indicators

**Goal**: Forecast GDP growth and inflation.

```
Step 1: search_tickers("GDP inflation")
  → GDP, GDPC1, CPIAUCSL, CPILFESL

Step 2: download_tickers(["GDPC1", "CPIAUCSL"], start="1990-01-01")

Step 3: analyze(dataset_name="gdpc1", target_col="value", mode="forecast")
  → ARIMA forecast on real GDP
  → Check forecast plot and confidence intervals

Step 4: analyze(dataset_name="cpiaucsl", target_col="value", mode="forecast")
  → ARIMA forecast on CPI
```

## Workflow E: Emerging Market Analysis

**Goal**: Analyze emerging market equity ETFs.

```
Step 1: list_universe(asset_class="EQUITY", sub_asset_class="Emerging")
  → IEMG, FXI, KWEB, INDA, EWZ, EWW, EWY, EWT

Step 2: download_tickers(["IEMG", "FXI", "INDA", "EWZ"], start="2015-01-01")

Step 3: For each:
    analyze(dataset_name="iemg", target_col="value", mode="recommend")
  → Runtime auto-selects best analysis based on data characteristics
```

## Common Patterns

### Pattern: Returns Analysis (Not Prices)

For volatility and GARCH analysis, the target should be returns, not raw prices. The downloaded data column is "value" (close price). Use analyze with mode="volatility" which handles this internally, or compute log returns via expert query:

```
delegate_to_expert("Use query_data to compute log returns from spy: 
  SELECT date, ln(value/lag(value) OVER (ORDER BY date)) AS ret 
  FROM dataset('spy') ORDER BY date")
```

### Pattern: Panel Data (Multiple Entities)

When downloading multiple tickers, each gets its own dataset. Compare by running analyze() on each separately, then use discover_analyses() to compare results.

### Pattern: Incremental Updates

Downloaded data is cached as parquet files. Running download_tickers again for the same ticker only fetches new data since the last download. Safe to call repeatedly.
