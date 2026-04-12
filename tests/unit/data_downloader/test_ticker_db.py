"""Tests for the ticker database and downloader tools."""

import pytest

from lazybridge.data_downloader.ticker_db import TickerDatabase, _parse_source
from lazybridge.data_downloader.schemas import TickerInfo, FetchResult, DownloaderConfig


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

class TestParseSource:
    def test_yahoo(self):
        assert _parse_source("Yahoo Finance (yfinance)") == "YAHOO"
        assert _parse_source("yahoo finance") == "YAHOO"
        assert _parse_source("finance.yahoo.com") == "YAHOO"

    def test_fred(self):
        assert _parse_source("https://fred.stlouisfed.org/series/DGS10") == "FRED"

    def test_ecb(self):
        assert _parse_source("https://data.ecb.europa.eu/data/key") == "ECB"
        assert _parse_source("https://ecb.europa.eu") == "ECB"

    def test_unknown(self):
        assert _parse_source("random source") == "UNKNOWN"
        assert _parse_source("") == "UNKNOWN"


# ---------------------------------------------------------------------------
# Ticker database
# ---------------------------------------------------------------------------

class TestTickerDatabase:
    @pytest.fixture
    def db(self):
        return TickerDatabase()

    def test_loads_all_tickers(self, db):
        assert len(db.list_all()) == 140

    def test_get_known_ticker(self, db):
        spy = db.get("SPY")
        assert spy is not None
        assert spy.ticker == "SPY"
        assert spy.source == "YAHOO"
        assert spy.asset_class == "EQUITY"

    def test_get_case_insensitive(self, db):
        assert db.get("spy") is not None
        assert db.get("SPY") is not None

    def test_get_unknown_ticker(self, db):
        assert db.get("NONEXISTENT") is None

    def test_search_by_ticker(self, db):
        results = db.search("SPY")
        assert any(t.ticker == "SPY" for t in results)

    def test_search_by_name(self, db):
        results = db.search("gold")
        assert len(results) > 0
        assert any("gold" in t.name.lower() for t in results)

    def test_search_by_sector(self, db):
        results = db.search("technology")
        assert len(results) > 0

    def test_filter_by_asset_class(self, db):
        equities = db.filter(asset_class="EQUITY")
        assert len(equities) > 30
        assert all(t.asset_class == "EQUITY" for t in equities)

    def test_filter_by_source(self, db):
        fred = db.filter(source="FRED")
        assert len(fred) > 20
        assert all(t.source == "FRED" for t in fred)

    def test_by_asset_class(self, db):
        groups = db.by_asset_class()
        assert "EQUITY" in groups
        assert "MACRO" in groups
        assert len(groups) >= 7

    def test_summary(self, db):
        s = db.summary()
        assert s["total_tickers"] == 140
        assert "EQUITY" in s["asset_classes"]
        assert "YAHOO" in s["sources"]

    def test_fred_tickers_detected(self, db):
        dgs10 = db.get("DGS10")
        assert dgs10 is not None
        assert dgs10.source == "FRED"

    def test_ecb_rates_sourced_from_fred(self, db):
        # ECB rate tickers (ECBDFR etc.) are sourced from FRED in this universe
        ecb_named = db.search("ecb")
        assert len(ecb_named) > 0
        assert all(t.source == "FRED" for t in ecb_named)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_ticker_info_defaults(self):
        ti = TickerInfo(ticker="SPY")
        assert ti.name == ""
        assert ti.source == ""

    def test_fetch_result_defaults(self):
        fr = FetchResult(ticker="SPY", source="YAHOO")
        assert fr.ok is False
        assert fr.registered_as is None

    def test_fetch_result_serialization(self):
        fr = FetchResult(
            ticker="SPY", source="YAHOO", ok=True,
            rows=100, date_start="2024-01-01", date_end="2024-04-01",
            file_path="/tmp/spy.parquet", frequency="daily",
            registered_as="spy",
        )
        data = fr.model_dump()
        restored = FetchResult(**data)
        assert restored.registered_as == "spy"

    def test_config_defaults(self):
        cfg = DownloaderConfig()
        assert cfg.default_start == "2000-01-01"
        assert cfg.max_retries == 3
