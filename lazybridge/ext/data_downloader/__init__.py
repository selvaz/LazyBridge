"""data_downloader — market data ingestion for LazyBridge stat_runtime.

Supports Yahoo Finance, FRED, and ECB data sources across a 140-ticker
universe organized in a 4-layer asset class taxonomy.

Quick start::

    from lazybridge.ext.data_downloader import TickerDatabase, DataDownloader

    db = TickerDatabase()
    dl = DataDownloader(cache_dir="./ticker_cache")

    # Browse universe
    print(db.summary())
    spy = db.get("SPY")

    # Download
    result = dl.fetch("SPY", "YAHOO", "2020-01-01", "2024-01-01")
    print(result.rows, result.file_path)
"""

from lazybridge.ext.data_downloader.schemas import (
    DownloaderConfig,
    FetchResult,
    TickerInfo,
)
from lazybridge.ext.data_downloader.ticker_db import TickerDatabase


def _lazy_import_downloader():
    from lazybridge.ext.data_downloader.downloader import DataDownloader
    return DataDownloader


def _lazy_import_tools():
    from lazybridge.ext.data_downloader.tools import downloader_tools
    return downloader_tools


def __getattr__(name: str):
    """Lazy import for heavy deps (pandas) — ticker-only usage works without them."""
    if name == "DataDownloader":
        return _lazy_import_downloader()
    if name == "downloader_tools":
        return _lazy_import_tools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_downloader_skills(output_root: str = "./generated_skills"):
    """Build BM25-indexed skill bundle from downloader docs.

    Returns dict with skill_dir path, or empty dict if docs missing.
    """
    from pathlib import Path
    from lazybridge.ext.tools.doc_skills import build_skill

    docs_dir = Path(__file__).parent / "skill_docs"
    if not docs_dir.exists() or not any(docs_dir.iterdir()):
        return {}

    return {
        "data_downloader": build_skill(
            [str(docs_dir)],
            "data-downloader-guide",
            output_root=output_root,
            description="Guide to using the data downloader: 140-ticker universe, "
                        "Yahoo/FRED/ECB sources, download workflows, ticker taxonomy.",
        ),
    }


def downloader_skill_tools(skill_dir_map: dict):
    """Create LazyTool wrappers for downloader skill bundles."""
    from lazybridge.ext.tools.doc_skills import skill_tool

    tools = []
    if "data_downloader" in skill_dir_map:
        tools.append(skill_tool(
            skill_dir_map["data_downloader"]["skill_dir"],
            name="data_downloader_guide",
            guidance="Query this to learn about available tickers, data sources, "
                     "asset class taxonomy, and download workflows.",
        ))
    return tools


__all__ = [
    "DownloaderConfig",
    "FetchResult",
    "TickerInfo",
    "TickerDatabase",
    "DataDownloader",
    "downloader_tools",
    "build_downloader_skills",
    "downloader_skill_tools",
]
