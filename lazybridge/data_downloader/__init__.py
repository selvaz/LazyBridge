"""data_downloader — market data ingestion for LazyBridge stat_runtime.

Supports Yahoo Finance, FRED, and ECB data sources across a 140-ticker
universe organized in a 4-layer asset class taxonomy.

Quick start::

    from lazybridge.data_downloader import TickerDatabase, DataDownloader

    db = TickerDatabase()
    dl = DataDownloader(cache_dir="./ticker_cache")

    # Browse universe
    print(db.summary())
    spy = db.get("SPY")

    # Download
    result = dl.fetch("SPY", "YAHOO", "2020-01-01", "2024-01-01")
    print(result.rows, result.file_path)
"""

from lazybridge.data_downloader.schemas import (
    DownloaderConfig,
    FetchResult,
    TickerInfo,
)
from lazybridge.data_downloader.ticker_db import TickerDatabase
from lazybridge.data_downloader.downloader import DataDownloader
from lazybridge.data_downloader.tools import downloader_tools

__all__ = [
    "DownloaderConfig",
    "FetchResult",
    "TickerInfo",
    "TickerDatabase",
    "DataDownloader",
    "downloader_tools",
]
