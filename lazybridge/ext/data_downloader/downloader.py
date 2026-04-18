"""Core data download logic — Yahoo Finance, FRED, ECB.

Ported from ts1_downloader/checks1_improved.py with adaptations for
LazyBridge integration (stat_runtime catalog registration, Pydantic results).
"""

from __future__ import annotations

import logging
import random
import re
import time
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from lazybridge.ext.data_downloader.schemas import (
    DownloaderConfig,
    FetchResult,
    TickerInfo,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _iso_today() -> str:
    return datetime.now(UTC).date().isoformat()


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _end_plus_days(iso_date: str, n: int) -> str:
    d = datetime.strptime(iso_date[:10], "%Y-%m-%d") + timedelta(days=n)
    return d.strftime("%Y-%m-%d")


def _guess_freq(df: pd.DataFrame) -> str:
    """Infer frequency from median gap between dates."""
    if df is None or len(df) < 3:
        return "unknown"
    dates = pd.to_datetime(df["date"]).sort_values()
    gaps = dates.diff().dropna().dt.days
    if gaps.empty:
        return "unknown"
    med = gaps.median()
    if med <= 1.5:
        return "daily"
    if med <= 8:
        return "weekly"
    if med <= 35:
        return "monthly"
    if med <= 100:
        return "quarterly"
    return "annual"


def _http_get(url: str, cfg: DownloaderConfig, params: dict | None = None):
    """HTTP GET with retry logic."""
    import requests  # type: ignore[import-untyped]

    for attempt in range(cfg.max_retries):
        try:
            r = requests.get(url, params=params, timeout=cfg.request_timeout)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            _logger.warning(
                "HTTP GET failed (attempt %d/%d) url=%s: %s: %s",
                attempt + 1,
                cfg.max_retries,
                url,
                type(exc).__name__,
                exc,
            )
            if attempt == cfg.max_retries - 1:
                raise
            # Exponential backoff with ±25% jitter — prevents a
            # thundering herd when many workers retry after the same
            # rate-limit window (audit L10).  Matches the jitter
            # formula used in lazybridge.core.executor.
            delay = cfg.retry_sleep * (2**attempt) * (0.75 + random.random() * 0.5)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Source-specific fetchers
# ---------------------------------------------------------------------------


def fetch_fred(series_id: str, start: str, end: str, cfg: DownloaderConfig) -> pd.DataFrame:
    """Download a FRED series (CSV, no API key needed)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = _http_get(url, cfg, params={"id": series_id})
    df = pd.read_csv(StringIO(r.text))
    if len(df.columns) < 2:
        return pd.DataFrame(columns=["date", "value"])
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[df["date"] >= pd.to_datetime(start)]
    df = df[df["date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def fetch_ecb(key_str: str, start: str, end: str, cfg: DownloaderConfig) -> pd.DataFrame:
    """Download an ECB Statistical Data Warehouse series (CSV)."""
    key = key_str.strip()
    if "/" in key:
        dataset, dkey = key.split("/", 1)
    else:
        parts = key.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"ECB key format not recognised: {key}")
        dataset, dkey = parts[0], parts[1]
    url = f"https://data-api.ecb.europa.eu/service/data/{dataset}/{dkey}"
    r = _http_get(url, cfg, params={"format": "csvdata"})
    raw = pd.read_csv(StringIO(r.text))
    if "TIME_PERIOD" not in raw.columns or "OBS_VALUE" not in raw.columns:
        raise ValueError(f"Unexpected ECB CSV columns: {raw.columns.tolist()[:10]}")
    df = raw[["TIME_PERIOD", "OBS_VALUE"]].rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df[df["date"] >= pd.to_datetime(start)]
    df = df[df["date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def fetch_yahoo_batch(
    tickers: list[str],
    start: str,
    end: str,
    cfg: DownloaderConfig,
) -> dict[str, pd.DataFrame]:
    """Download multiple Yahoo Finance tickers in a single call."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance") from None

    end_query = _end_plus_days(end, cfg.yahoo_end_plus_days)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_query,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    results: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        for t in tickers:
            results[t] = pd.DataFrame(columns=["date", "value"])
        return results

    # Extract Close / Adj Close
    if isinstance(raw.columns, pd.MultiIndex):
        fields = raw.columns.get_level_values(0).unique().tolist()
        pick = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else fields[0])
        prices = raw[pick].copy()
    else:
        prices = raw.copy()
        if "Adj Close" in prices.columns:
            prices = prices[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        elif "Close" in prices.columns:
            prices = prices[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            nums = [c for c in prices.columns if pd.api.types.is_numeric_dtype(prices[c])]
            if not nums:
                for t in tickers:
                    results[t] = pd.DataFrame(columns=["date", "value"])
                return results
            prices = prices[[nums[0]]].rename(columns={nums[0]: tickers[0]})

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.sort_index(inplace=True)

    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")

    for t in tickers:
        if t not in prices.columns:
            results[t] = pd.DataFrame(columns=["date", "value"])
            continue
        s = prices[t].dropna()
        df = s.reset_index()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if not pd.isna(start_dt):
            df = df[df["date"] >= start_dt]
        if not pd.isna(end_dt):
            df = df[df["date"].dt.date <= end_dt.date()]
        results[t] = df.reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class CacheManager:
    """Parquet-based per-ticker cache with merge/dedup."""

    def __init__(self, cache_dir: str) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, ticker: str) -> str:
        return str(self._dir / f"{_safe_filename(ticker)}.parquet")

    def load(self, ticker: str) -> pd.DataFrame | None:
        p = Path(self.path_for(ticker))
        if not p.exists():
            return None
        try:
            return pd.read_parquet(p)
        except Exception as exc:
            # Don't crash the whole batch on one bad parquet file — but
            # leave a breadcrumb at DEBUG so operators can diagnose the
            # root cause (corrupt / partial write / schema drift).
            _logger.debug(
                "load(%r): pd.read_parquet(%s) failed — %s: %s",
                ticker,
                p,
                type(exc).__name__,
                exc,
            )
            return None

    def save(self, ticker: str, df: pd.DataFrame) -> str:
        path = self.path_for(ticker)
        df.to_parquet(path, index=False)
        return path

    @staticmethod
    def merge_dedup(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
        if existing is None or existing.empty:
            df = new.copy()
        else:
            df = pd.concat([existing, new], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        return df.reset_index(drop=True)

    @staticmethod
    def last_date(df: pd.DataFrame | None) -> pd.Timestamp | None:
        if df is None or df.empty:
            return None
        return pd.to_datetime(df["date"]).max()


# ---------------------------------------------------------------------------
# Main downloader class
# ---------------------------------------------------------------------------


class DataDownloader:
    """Downloads market data from Yahoo/FRED/ECB with parquet caching."""

    def __init__(
        self,
        cache_dir: str = "ticker_cache",
        config: DownloaderConfig | None = None,
    ) -> None:
        self.config = config or DownloaderConfig()
        self.cache = CacheManager(cache_dir)

    def fetch(
        self,
        ticker: str,
        source: str,
        start: str | None = None,
        end: str | None = None,
        source_url: str = "",
    ) -> FetchResult:
        """Download a single ticker. Uses cache for incremental updates."""
        start = start or self.config.default_start
        end = end or _iso_today()
        cfg = self.config

        try:
            # Load cache for incremental
            existing = self.cache.load(ticker)
            last = CacheManager.last_date(existing)

            # Effective start: day after last cached date
            eff_start = start
            if existing is not None and last is not None:
                eff_start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
                if eff_start > end:
                    # Already up to date
                    path = self.cache.path_for(ticker)
                    return FetchResult(
                        ticker=ticker,
                        source=source,
                        ok=True,
                        rows=len(existing),
                        cached=True,
                        date_start=str(existing["date"].min().date()),
                        date_end=str(existing["date"].max().date()),
                        file_path=path,
                        frequency=_guess_freq(existing),
                    )

            # Download
            src = source.upper()
            if src == "YAHOO":
                batch = fetch_yahoo_batch([ticker], eff_start, end, cfg)
                new_df = batch.get(ticker, pd.DataFrame(columns=["date", "value"]))
            elif src == "FRED":
                new_df = fetch_fred(ticker, eff_start, end, cfg)
            elif src == "ECB":
                ecb_key = source_url.split("/")[-1] if "/" in source_url else ticker
                new_df = fetch_ecb(ecb_key, eff_start, end, cfg)
            else:
                # Try as Yahoo by default
                batch = fetch_yahoo_batch([ticker], eff_start, end, cfg)
                new_df = batch.get(ticker, pd.DataFrame(columns=["date", "value"]))

            # Merge with cache
            merged = CacheManager.merge_dedup(existing, new_df)
            path = self.cache.save(ticker, merged)

            return FetchResult(
                ticker=ticker,
                source=source,
                ok=True,
                rows=len(merged),
                date_start=str(merged["date"].min().date()) if len(merged) > 0 else None,
                date_end=str(merged["date"].max().date()) if len(merged) > 0 else None,
                file_path=path,
                frequency=_guess_freq(merged),
                cached=existing is not None and len(existing) > 0,
            )

        except Exception as exc:
            _logger.warning("Failed to fetch %s: %s", ticker, exc)
            return FetchResult(
                ticker=ticker,
                source=source,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def fetch_batch(
        self,
        tickers: list[TickerInfo],
        start: str | None = None,
        end: str | None = None,
    ) -> list[FetchResult]:
        """Download multiple tickers, batching Yahoo calls for efficiency."""
        start = start or self.config.default_start
        end = end or _iso_today()
        cfg = self.config

        # Separate by source
        yahoo_tickers = [t for t in tickers if t.source == "YAHOO"]
        other_tickers = [t for t in tickers if t.source != "YAHOO"]

        results: list[FetchResult] = []

        # Batch Yahoo download
        if yahoo_tickers:
            symbols = [t.ticker for t in yahoo_tickers]
            try:
                yahoo_data = fetch_yahoo_batch(symbols, start, end, cfg)
            except Exception as exc:
                _logger.warning("Yahoo batch failed: %s", exc)
                yahoo_data = {}

            for ti in yahoo_tickers:
                new_df = yahoo_data.get(ti.ticker, pd.DataFrame(columns=["date", "value"]))
                existing = self.cache.load(ti.ticker)
                try:
                    merged = CacheManager.merge_dedup(existing, new_df)
                    path = self.cache.save(ti.ticker, merged)
                    results.append(
                        FetchResult(
                            ticker=ti.ticker,
                            source="YAHOO",
                            ok=True,
                            rows=len(merged),
                            date_start=str(merged["date"].min().date()) if len(merged) > 0 else None,
                            date_end=str(merged["date"].max().date()) if len(merged) > 0 else None,
                            file_path=path,
                            frequency=_guess_freq(merged),
                        )
                    )
                except Exception as exc:
                    results.append(
                        FetchResult(
                            ticker=ti.ticker,
                            source="YAHOO",
                            ok=False,
                            error=str(exc),
                        )
                    )

        # FRED/ECB one by one
        for ti in other_tickers:
            r = self.fetch(ti.ticker, ti.source, start, end, source_url=ti.source_url)
            results.append(r)

        return results

    def download_and_register(
        self,
        ticker_infos: list[TickerInfo],
        start: str | None = None,
        end: str | None = None,
        runtime: Any = None,
    ) -> list[FetchResult]:
        """Download tickers and register each in stat_runtime catalog."""
        results = self.fetch_batch(ticker_infos, start, end)

        if runtime is None:
            return results

        for r in results:
            if not r.ok or not r.file_path:
                continue
            try:
                ti = next((t for t in ticker_infos if t.ticker == r.ticker), None)
                dataset_name = _safe_filename(r.ticker).lower()

                # Detect if it's value-only (FRED/macro) or OHLCV
                freq_map = {
                    "daily": "daily",
                    "weekly": "weekly",
                    "monthly": "monthly",
                    "quarterly": "quarterly",
                    "annual": "annual",
                }
                freq = freq_map.get(r.frequency, "daily")

                desc = ti.name if ti else f"{r.ticker} ({r.source})"

                runtime.catalog.register_parquet(
                    name=dataset_name,
                    uri=r.file_path,
                    time_column="date",
                    frequency=freq,
                    business_description=desc,
                )
                r.registered_as = dataset_name
            except Exception as exc:
                _logger.warning("Failed to register %s: %s", r.ticker, exc)

        return results
