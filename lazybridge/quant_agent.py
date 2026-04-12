"""Pre-configured Quantitative Analysis Agent.

A specialized LazyAgent with all data download + statistical analysis tools
and skills wired together. This is the zero-boilerplate entry point for
LLM-driven financial analysis.

Usage::

    from lazybridge.quant_agent import quant_agent

    agent, rt = quant_agent("anthropic")
    resp = agent.loop("Download SPY, AAPL, and MSFT. Analyze their volatility.")
    print(resp.content)
    rt.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

QUANT_SYSTEM_PROMPT = """\
You are a quantitative financial analyst with access to a 140-ticker universe \
(equities, bonds, commodities, macro, FX, crypto, real estate) and a full \
statistical analysis runtime.

## Your Tools

**Data Discovery & Download:**
- `list_universe(asset_class=...)` — browse 140 tickers by asset class
- `search_tickers(query)` — search by name, symbol, sector, country
- `download_tickers(tickers, start, end)` — download from Yahoo/FRED/ECB and register

**Analysis:**
- `discover_data()` — see registered datasets with column roles and quality signals
- `analyze(dataset, target_col, mode)` — run goal-oriented analysis:
  - mode="recommend" — auto-select best analysis
  - mode="volatility" — GARCH volatility modeling
  - mode="forecast" — ARIMA time-series forecast
  - mode="regime" — Markov regime detection
  - mode="describe" — descriptive statistics
- `discover_analyses()` — review completed analyses with metrics and plots

**Knowledge:**
- `data_downloader_guide(query)` — look up ticker info, data sources, workflows
- `stat_tool_guide(query)` — look up tool usage, parameters, error recovery
- `quant_methodology(query)` — look up statistical methods and best practices

## Workflow

1. **Find tickers**: Use search_tickers or list_universe
2. **Download data**: Use download_tickers (auto-registers in runtime)
3. **Discover**: Use discover_data to verify what's available
4. **Analyze**: Use analyze with appropriate mode
5. **Report**: Present interpretation, model adequacy, plots, and next steps

## Important Rules

- Downloaded data has columns: date (Date), value (Float64)
- Use target_col="value" for analysis (unless dataset has other numeric columns)
- For panel data, each ticker becomes its own dataset (lowercase name)
- Always check model_adequate before trusting results
- Reference artifact paths so the user can view generated plots
- Use the skill tools when unsure about methodology or tool parameters
- If analysis fails, suggest alternative approaches from next_steps
"""


def quant_agent(
    provider: str = "anthropic",
    *,
    model: str | None = None,
    artifacts_dir: str = "artifacts",
    cache_dir: str | None = None,
    name: str = "quant_analyst",
    system: str | None = None,
    **agent_kwargs: Any,
) -> tuple[Any, Any]:
    """Create a fully-equipped quantitative analysis agent.

    Includes:
    - Data download tools (list_universe, search_tickers, download_tickers)
    - Statistical analysis tools (discover_data, discover_analyses, analyze, register_dataset)
    - Expert delegation (delegate_to_expert for low-level access)
    - BM25 skill retrieval (downloader guide, stat tool guide, quant methodology)

    Args:
        provider: LLM provider ("anthropic", "openai", "google", "deepseek")
        model: Model override (defaults to provider's standard)
        artifacts_dir: Where to store analysis artifacts and ticker cache
        cache_dir: Ticker data cache (defaults to {artifacts_dir}/ticker_cache)
        name: Agent name
        system: System prompt override (defaults to QUANT_SYSTEM_PROMPT)

    Returns:
        (agent, runtime) tuple. Call rt.close() when done.

    Usage::

        agent, rt = quant_agent("anthropic")
        resp = agent.loop("Download SPY and analyze its volatility")
        print(resp.content)
        rt.close()
    """
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.lazy_tool import LazyTool
    from lazybridge.stat_runtime.runner import StatRuntime
    from lazybridge.stat_runtime.tools import (
        stat_tools,
        build_stat_skills,
        stat_skill_tools,
    )
    from lazybridge.data_downloader import (
        TickerDatabase,
        DataDownloader,
        downloader_tools,
        build_downloader_skills,
        downloader_skill_tools,
    )

    # Runtime
    rt = StatRuntime(artifacts_dir=artifacts_dir)
    effective_cache = cache_dir or str(Path(artifacts_dir) / "ticker_cache")

    # Downloader
    db = TickerDatabase()
    dl = DataDownloader(cache_dir=effective_cache)

    # Tools: downloader + high-level stat
    all_tools: list[LazyTool] = []
    all_tools.extend(downloader_tools(rt, db, dl))
    all_tools.extend(stat_tools(rt, level="high"))

    # Expert sub-agent for low-level access
    try:
        expert = LazyAgent(
            provider,
            model=model,
            name="stat_expert",
            system="You are an expert statistical analyst. Use the available tools "
                   "to fulfill the task precisely. Return results as structured data.",
            tools=stat_tools(rt, level="low"),
            **agent_kwargs,
        )
        expert_tool = expert.as_tool(
            name="delegate_to_expert",
            description="Delegate to expert agent for custom SQL, specific model params, or manual control",
            guidance="Use when the user needs: specific ARIMA(p,d,q) orders, custom SQL queries, "
                     "individual diagnostic tests, or manual plot retrieval.",
        )
        all_tools.append(expert_tool)
    except Exception:
        _logger.warning("Expert sub-agent not created", exc_info=True)

    # Skills: downloader guide + stat guide + quant methodology
    try:
        dl_skills = build_downloader_skills(
            output_root=str(Path(artifacts_dir) / "skills"),
        )
        all_tools.extend(downloader_skill_tools(dl_skills))
    except Exception:
        _logger.debug("Downloader skills not loaded", exc_info=True)

    try:
        stat_skills = build_stat_skills(
            output_root=str(Path(artifacts_dir) / "skills"),
        )
        all_tools.extend(stat_skill_tools(stat_skills))
    except Exception:
        _logger.debug("Stat skills not loaded", exc_info=True)

    # Agent
    agent = LazyAgent(
        provider,
        model=model,
        name=name,
        description="Quantitative analyst with data download and statistical analysis capabilities",
        system=system or QUANT_SYSTEM_PROMPT,
        tools=all_tools,
        **agent_kwargs,
    )

    _logger.info(
        "Created quant_agent with %d tools (%d tickers in universe)",
        len(all_tools), len(db.list_all()),
    )

    return agent, rt
