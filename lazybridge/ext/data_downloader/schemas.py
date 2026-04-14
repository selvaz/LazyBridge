"""Pydantic models for the data downloader module."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DownloaderConfig(BaseModel):
    """Configuration for data downloads."""
    default_start: str = "2000-01-01"
    request_timeout: int = 30
    max_retries: int = 3
    retry_sleep: float = 2.0
    yahoo_end_plus_days: int = 1  # yfinance end is exclusive


class TickerInfo(BaseModel):
    """Metadata for a ticker in the universe database."""
    ticker: str
    name: str = ""
    source: str = ""               # YAHOO, FRED, ECB, UNKNOWN
    source_url: str = ""           # raw Fonte value
    area: str = ""
    asset_class: str = ""          # Layer1_AssetClass
    benchmark: str = ""            # Layer1_Benchmark
    sub_asset_class: str = ""      # Layer2_SubAssetClass
    geographic_sector: str = ""    # Layer3_Geographic_Sector
    granular: str = ""             # Layer4_Granular


class FetchResult(BaseModel):
    """Result of downloading a single ticker."""
    ticker: str
    source: str
    ok: bool = False
    rows: int = 0
    date_start: str | None = None
    date_end: str | None = None
    file_path: str = ""
    frequency: str = ""
    error: str | None = None
    cached: bool = False
    registered_as: str | None = Field(
        default=None,
        description="Dataset name in stat_runtime if auto-registered",
    )
