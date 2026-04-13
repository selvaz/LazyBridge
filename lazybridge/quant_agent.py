"""Pre-configured Quantitative Analysis Agent.

A specialized LazyAgent with N sub-agent pipeline tools — each function gets
its own agent(output_schema) → function chain.  The router agent knows the
methodology (via skill tools) and calls the right sub-agent as a tool.

Architecture::

    router_agent (skills: methodology, tool guide, downloader guide)
    ├── discover_data          (plain tool — no params)
    ├── analyze_agent          (sub-agent → analyze())
    ├── discover_analyses_agent(sub-agent → discover_analyses())
    ├── register_dataset_agent (sub-agent → register_dataset())
    ├── list_universe_agent    (sub-agent → list_universe())
    ├── search_tickers_agent   (sub-agent → search_tickers())
    ├── download_tickers_agent (sub-agent → download_tickers())
    ├── fit_model_agent        (sub-agent → fit_model())
    ├── query_data_agent       (sub-agent → query_data())
    ├── forecast_agent         (sub-agent → forecast_model())
    ├── diagnostics_agent      (sub-agent → run_diagnostics())
    ├── compare_models_agent   (sub-agent → compare_models())
    ├── profile_dataset_agent  (sub-agent → profile_dataset())
    ├── get_run_agent          (sub-agent → get_run())
    ├── list_runs_agent        (sub-agent → list_runs())
    ├── list_artifacts_agent   (sub-agent → list_artifacts())
    └── get_plot_agent         (sub-agent → get_plot())

Each sub-agent pipeline:
  1. Receives {"task": "natural language request"} from the router
  2. Sub-agent (LLM) produces structured output = function's input schema
  3. Chain passes model_dump() to the function for deterministic execution
  4. Function result returns to the router

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

Each tool accepts a natural language task and handles parameter extraction \
automatically. Just describe what you need.

**Data Discovery & Download:**
- `list_universe` — browse 140 tickers by asset class
- `search_tickers` — search by name, symbol, sector, country
- `download_tickers` — download from Yahoo/FRED/ECB and register

**Discovery:**
- `discover_data` — see registered datasets with column roles and quality signals
- `discover_analyses` — review completed analyses with metrics and plots

**Analysis:**
- `analyze` — run goal-oriented analysis:
  - mode="recommend" — auto-select best analysis
  - mode="volatility" — GARCH volatility modeling
  - mode="forecast" — ARIMA time-series forecast
  - mode="regime" — Markov regime detection
  - mode="describe" — descriptive statistics
- `register_dataset` — register a new data file for analysis

**Expert Tools:**
- `fit_model` — fit a specific model with custom parameters
- `query_data` — run SQL on registered datasets
- `forecast_model` — generate forecast from a fitted model
- `run_diagnostics` — stationarity tests on a data column
- `compare_models` — compare multiple model runs
- `profile_dataset` — column-level statistics
- `get_run` — retrieve a past model run
- `list_runs` — list past model runs
- `list_artifacts` — list artifacts for a run
- `get_plot` — get a specific plot

**Knowledge:**
- `data_downloader_guide` — look up ticker info, data sources, workflows
- `stat_tool_guide` — look up tool usage, parameters, error recovery
- `quant_methodology` — look up statistical methods and best practices

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

    Uses N specialized sub-agent pipelines — each function gets its own
    agent(output_schema) → function chain.  The router agent knows the
    methodology and calls the right sub-agent as a tool.

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
    from lazybridge.stat_runtime.schemas import (
        AnalyzeInput,
        CompareModelsInput,
        DiscoverAnalysesInput,
        DownloadTickersInput,
        FitModelInput,
        ForecastInput,
        GetPlotInput,
        GetRunInput,
        ListArtifactsInput,
        ListRunsInput,
        ListUniverseInput,
        ProfileDatasetInput,
        QueryDataInput,
        RegisterDatasetInput,
        RunDiagnosticsInput,
        SearchTickersInput,
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

    # Get raw tool functions (bound to runtime via closures)
    all_stat = stat_tools(rt, level="all")
    dl_tools = downloader_tools(rt, db, dl)

    # Build name→LazyTool lookup for the raw functions
    raw_tools = {t.name: t for t in all_stat + dl_tools}

    # Common agent kwargs for sub-agents
    sub_kwargs = {k: v for k, v in agent_kwargs.items() if k not in ("tools",)}

    # ---------------------------------------------------------------
    # Build agent_tool pipelines for each function
    # ---------------------------------------------------------------

    def _build_agent_tool(
        tool_name: str,
        input_schema: type,
        description: str,
        guidance: str | None = None,
    ) -> LazyTool:
        """Build a sub-agent → function pipeline for a single tool."""
        raw = raw_tools[tool_name]
        return LazyTool.agent_tool(
            raw.func,
            input_schema=input_schema,
            provider=provider,
            model=model,
            name=tool_name,
            description=description,
            guidance=guidance,
            **sub_kwargs,
        )

    all_tools: list[LazyTool] = []

    # discover_data has no params — keep as plain tool
    if "discover_data" in raw_tools:
        all_tools.append(raw_tools["discover_data"])

    # High-level analysis tools
    all_tools.append(_build_agent_tool(
        "analyze", AnalyzeInput,
        "Run a goal-oriented analysis with automatic model selection",
        guidance="Primary analysis tool. Describe the analysis goal and the dataset. "
                 "Modes: recommend, describe, forecast, volatility, regime.",
    ))
    all_tools.append(_build_agent_tool(
        "discover_analyses", DiscoverAnalysesInput,
        "Discover completed analysis runs with metrics and artifact catalogs",
        guidance="Call to review what analyses exist. Optionally filter by dataset name.",
    ))
    all_tools.append(_build_agent_tool(
        "register_dataset", RegisterDatasetInput,
        "Register a Parquet or CSV file as a named dataset for analysis",
        guidance="Provide file path and metadata. After registering, call discover_data().",
    ))

    # Data downloader tools
    all_tools.append(_build_agent_tool(
        "list_universe", ListUniverseInput,
        "Browse the 140-ticker universe by asset class",
        guidance="Describe what asset classes or categories you want to explore.",
    ))
    all_tools.append(_build_agent_tool(
        "search_tickers", SearchTickersInput,
        "Search tickers by name, symbol, sector, or country",
        guidance="Describe what you're looking for — partial matches work.",
    ))
    all_tools.append(_build_agent_tool(
        "download_tickers", DownloadTickersInput,
        "Download market data and register in stat_runtime for analysis",
        guidance="List the tickers to download. Sources auto-detected from universe.",
    ))

    # Expert-level stat tools
    all_tools.append(_build_agent_tool(
        "fit_model", FitModelInput,
        "Fit a specific statistical model (OLS, ARIMA, GARCH, Markov) to data",
        guidance="Specify model family, target column, dataset, and any custom parameters.",
    ))
    all_tools.append(_build_agent_tool(
        "query_data", QueryDataInput,
        "Run a SQL query on registered datasets using dataset('name') macro",
        guidance="Describe the SQL query needed. Only SELECT statements allowed.",
    ))
    all_tools.append(_build_agent_tool(
        "forecast_model", ForecastInput,
        "Generate a forecast from a previously fitted model",
        guidance="Provide run_id and number of forecast steps.",
    ))
    all_tools.append(_build_agent_tool(
        "run_diagnostics", RunDiagnosticsInput,
        "Run stationarity tests (ADF + KPSS) on a data column",
        guidance="Specify dataset name and column to test.",
    ))
    all_tools.append(_build_agent_tool(
        "compare_models", CompareModelsInput,
        "Compare multiple model runs by AIC, BIC, and other metrics",
        guidance="Provide a list of run IDs to compare.",
    ))
    all_tools.append(_build_agent_tool(
        "profile_dataset", ProfileDatasetInput,
        "Compute column-level statistics for a registered dataset",
        guidance="Provide the dataset name to profile.",
    ))
    all_tools.append(_build_agent_tool(
        "get_run", GetRunInput,
        "Retrieve a past model run with full metrics and artifact paths",
        guidance="Provide the run ID to retrieve.",
    ))
    all_tools.append(_build_agent_tool(
        "list_runs", ListRunsInput,
        "List past model runs, optionally filtered by dataset",
        guidance="Optionally specify dataset name and limit.",
    ))
    all_tools.append(_build_agent_tool(
        "list_artifacts", ListArtifactsInput,
        "List all artifacts (plots, data, summaries) for a model run",
        guidance="Provide run ID and optionally filter by artifact type.",
    ))
    all_tools.append(_build_agent_tool(
        "get_plot", GetPlotInput,
        "Get the file path for a specific plot from a model run",
        guidance="Provide run ID and plot name (residuals, volatility, forecast, regimes).",
    ))

    # Skills: downloader guide + stat guide + quant methodology
    try:
        dl_skills = build_downloader_skills(
            output_root=str(Path(artifacts_dir) / "skills"),
        )
        all_tools.extend(downloader_skill_tools(dl_skills))
    except Exception:
        _logger.debug("Downloader skills not loaded", exc_info=True)

    try:
        stat_skills_dirs = build_stat_skills(
            output_root=str(Path(artifacts_dir) / "skills"),
        )
        all_tools.extend(stat_skill_tools(stat_skills_dirs))
    except Exception:
        _logger.debug("Stat skills not loaded", exc_info=True)

    # Router agent
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
