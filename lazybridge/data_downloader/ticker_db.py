"""Ticker universe database — 140 tickers across 7 asset classes.

Loads the bundled CSV and provides search/filter/lookup capabilities.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from lazybridge.data_downloader.schemas import TickerInfo

_logger = logging.getLogger(__name__)

_DEFAULT_CSV = Path(__file__).parent / "data" / "ticker_universe.csv"

_SOURCE_PATTERNS = {
    "fred.stlouisfed.org": "FRED",
    "data.ecb.europa.eu": "ECB",
    "ecb.europa.eu": "ECB",
    "yahoo finance": "YAHOO",
    "yfinance": "YAHOO",
    "finance.yahoo.com": "YAHOO",
}


def _parse_source(fonte: str) -> str:
    """Detect data source from the Fonte column."""
    f = (fonte or "").strip().lower()
    for pattern, source in _SOURCE_PATTERNS.items():
        if pattern in f:
            return source
    return "UNKNOWN"


class TickerDatabase:
    """Queryable database of the ticker universe."""

    def __init__(self, csv_path: str | None = None) -> None:
        path = Path(csv_path) if csv_path else _DEFAULT_CSV
        self._tickers: list[TickerInfo] = []
        self._by_ticker: dict[str, TickerInfo] = {}
        self._load(path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            _logger.warning("Ticker universe not found: %s", path)
            return
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                info = TickerInfo(
                    ticker=row.get("Ticker", "").strip(),
                    name=row.get("Serie", "").strip(),
                    source=_parse_source(row.get("Fonte", "")),
                    source_url=row.get("Fonte", "").strip(),
                    area=row.get("Area", "").strip(),
                    asset_class=row.get("Layer1_AssetClass", "").strip(),
                    benchmark=row.get("Layer1_Benchmark", "").strip(),
                    sub_asset_class=row.get("Layer2_SubAssetClass", "").strip(),
                    geographic_sector=row.get("Layer3_Geographic_Sector", "").strip(),
                    granular=row.get("Layer4_Granular", "").strip(),
                )
                if info.ticker:
                    self._tickers.append(info)
                    self._by_ticker[info.ticker.upper()] = info
        _logger.info("Loaded %d tickers from %s", len(self._tickers), path)

    def list_all(self) -> list[TickerInfo]:
        """Return all tickers."""
        return list(self._tickers)

    def get(self, ticker: str) -> TickerInfo | None:
        """Look up a ticker by symbol (case-insensitive)."""
        return self._by_ticker.get(ticker.upper())

    def search(self, query: str) -> list[TickerInfo]:
        """Search tickers by name, symbol, or any taxonomy field."""
        q = query.lower()
        return [
            t for t in self._tickers
            if q in t.ticker.lower()
            or q in t.name.lower()
            or q in t.asset_class.lower()
            or q in t.sub_asset_class.lower()
            or q in t.geographic_sector.lower()
            or q in t.granular.lower()
            or q in t.area.lower()
        ]

    def filter(
        self,
        asset_class: str | None = None,
        sub_asset_class: str | None = None,
        geographic_sector: str | None = None,
        source: str | None = None,
    ) -> list[TickerInfo]:
        """Filter tickers by taxonomy fields."""
        results = self._tickers
        if asset_class:
            ac = asset_class.upper()
            results = [t for t in results if t.asset_class.upper() == ac]
        if sub_asset_class:
            sa = sub_asset_class.lower()
            results = [t for t in results if sa in t.sub_asset_class.lower()]
        if geographic_sector:
            gs = geographic_sector.lower()
            results = [t for t in results if gs in t.geographic_sector.lower()]
        if source:
            src = source.upper()
            results = [t for t in results if t.source == src]
        return results

    def by_asset_class(self) -> dict[str, list[TickerInfo]]:
        """Group tickers by Layer1 asset class."""
        groups: dict[str, list[TickerInfo]] = {}
        for t in self._tickers:
            groups.setdefault(t.asset_class, []).append(t)
        return groups

    def summary(self) -> dict[str, Any]:
        """Compact summary for LLM consumption."""
        groups = self.by_asset_class()
        return {
            "total_tickers": len(self._tickers),
            "asset_classes": {
                ac: {
                    "count": len(tickers),
                    "tickers": [t.ticker for t in tickers],
                }
                for ac, tickers in sorted(groups.items())
            },
            "sources": {
                src: sum(1 for t in self._tickers if t.source == src)
                for src in sorted({t.source for t in self._tickers})
            },
        }
