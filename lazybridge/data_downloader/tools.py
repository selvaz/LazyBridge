"""LLM-callable tools for the data downloader.

Three tools for the agent:
- list_universe: browse the 140-ticker universe by asset class
- search_tickers: search tickers by name/symbol/category
- download_tickers: download data and auto-register in stat_runtime
"""

from __future__ import annotations

from typing import Annotated, Any

from lazybridge.lazy_tool import LazyTool
from lazybridge.data_downloader.downloader import DataDownloader
from lazybridge.data_downloader.schemas import FetchResult
from lazybridge.data_downloader.ticker_db import TickerDatabase


def _error(exc: Exception) -> dict:
    return {"error": True, "type": type(exc).__name__, "message": str(exc)}


def downloader_tools(
    runtime: Any,
    ticker_db: TickerDatabase,
    downloader: DataDownloader,
) -> list[LazyTool]:
    """Create LLM-callable tools for data download and universe browsing."""

    db = ticker_db
    dl = downloader
    rt = runtime

    # -- list_universe ---------------------------------------------------

    def list_universe(
        asset_class: Annotated[str | None, "Filter by asset class: EQUITY, FIXED_INCOME, COMMODITIES, REAL_ESTATE, ALTERNATIVES, MACRO, FX"] = None,
        sub_asset_class: Annotated[str | None, "Filter by sub-asset class (e.g. 'Developed', 'Government', 'Energy')"] = None,
    ) -> dict[str, Any]:
        """List available tickers from the 140-ticker universe database.

        Returns tickers grouped by asset class with counts.
        Filter by asset_class or sub_asset_class to narrow results.
        """
        try:
            if asset_class or sub_asset_class:
                tickers = db.filter(
                    asset_class=asset_class,
                    sub_asset_class=sub_asset_class,
                )
                return {
                    "total": len(tickers),
                    "filters_applied": {
                        k: v for k, v in [
                            ("asset_class", asset_class),
                            ("sub_asset_class", sub_asset_class),
                        ] if v
                    },
                    "tickers": [
                        {
                            "ticker": t.ticker,
                            "name": t.name,
                            "source": t.source,
                            "asset_class": t.asset_class,
                            "sub_asset": t.sub_asset_class,
                            "geo_sector": t.geographic_sector,
                            "granular": t.granular,
                        }
                        for t in tickers
                    ],
                }
            else:
                return db.summary()
        except Exception as exc:
            return _error(exc)

    # -- search_tickers --------------------------------------------------

    def search_tickers(
        query: Annotated[str, "Search query: ticker symbol, name, asset class, sector, or country"],
    ) -> dict[str, Any]:
        """Search the ticker universe by any field.

        Examples: 'SPY', 'gold', 'US equity', 'emerging', 'technology', 'treasury'
        """
        try:
            results = db.search(query)
            return {
                "query": query,
                "matches": len(results),
                "tickers": [
                    {
                        "ticker": t.ticker,
                        "name": t.name,
                        "source": t.source,
                        "asset_class": t.asset_class,
                        "sub_asset": t.sub_asset_class,
                        "geo_sector": t.geographic_sector,
                    }
                    for t in results[:30]  # cap for LLM context
                ],
            }
        except Exception as exc:
            return _error(exc)

    # -- download_tickers ------------------------------------------------

    def download_tickers(
        tickers: Annotated[list[str], "Ticker symbols to download (e.g. ['SPY', 'AAPL', 'DGS10'])"],
        start: Annotated[str | None, "Start date YYYY-MM-DD (default: 2000-01-01)"] = None,
        end: Annotated[str | None, "End date YYYY-MM-DD (default: today)"] = None,
        auto_register: Annotated[bool, "Auto-register downloaded data in stat_runtime"] = True,
    ) -> dict[str, Any]:
        """Download market data for tickers and optionally register in stat_runtime.

        Sources are auto-detected from the ticker universe:
        - Yahoo Finance: equities, ETFs, crypto, FX
        - FRED: US macro indicators (treasury rates, CPI, GDP, unemployment)
        - ECB: European economic data

        Tickers not in the universe are attempted via Yahoo Finance.
        Downloaded data is cached as parquet files for incremental updates.
        """
        try:
            # Resolve ticker info from universe
            ticker_infos = []
            unknown = []
            for sym in tickers:
                info = db.get(sym)
                if info:
                    ticker_infos.append(info)
                else:
                    # Not in universe — default to Yahoo
                    from lazybridge.data_downloader.schemas import TickerInfo
                    ticker_infos.append(TickerInfo(
                        ticker=sym, name=sym, source="YAHOO",
                        source_url="Yahoo Finance",
                    ))
                    unknown.append(sym)

            # Download and optionally register
            if auto_register and rt:
                results = dl.download_and_register(ticker_infos, start, end, rt)
            else:
                results = dl.fetch_batch(ticker_infos, start, end)

            # Build summary
            ok = [r for r in results if r.ok]
            failed = [r for r in results if not r.ok]

            summary: dict[str, Any] = {
                "total": len(results),
                "succeeded": len(ok),
                "failed": len(failed),
                "results": [r.model_dump() for r in results],
            }
            if unknown:
                summary["note"] = (
                    f"Tickers not in universe (defaulted to Yahoo): {unknown}"
                )
            if auto_register and ok:
                summary["registered_datasets"] = [
                    r.registered_as for r in ok if r.registered_as
                ]

            return summary
        except Exception as exc:
            return _error(exc)

    # -- Assemble --------------------------------------------------------

    return [
        LazyTool.from_function(
            list_universe, name="list_universe",
            description="Browse the 140-ticker universe by asset class",
            guidance="Call to see what tickers are available for download. "
                     "Filter by asset_class (EQUITY, FIXED_INCOME, COMMODITIES, "
                     "REAL_ESTATE, ALTERNATIVES, MACRO, FX) or sub_asset_class.",
        ),
        LazyTool.from_function(
            search_tickers, name="search_tickers",
            description="Search tickers by name, symbol, sector, or country",
            guidance="Use to find specific tickers. Supports partial matches: "
                     "'gold', 'treasury', 'emerging', 'technology', etc.",
        ),
        LazyTool.from_function(
            download_tickers, name="download_tickers",
            description="Download market data and register in stat_runtime for analysis",
            guidance="Provide ticker symbols. Sources auto-detected from universe. "
                     "Data is cached incrementally. Set register=True (default) to "
                     "auto-register for use with discover_data() and analyze().",
        ),
    ]
