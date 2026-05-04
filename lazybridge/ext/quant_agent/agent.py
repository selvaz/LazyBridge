"""Pre-configured Quantitative Analysis Agent.

Hybrid architecture: complex tools get dedicated sub-agent pipelines
(agent_tool) for intelligent parameter construction; simple tools use
direct tool calling (plain Tool) for efficiency.

Architecture::

    router_agent (skills: methodology, tool guide, downloader guide)
    │
    │ agent_tool pipelines (2x LLM calls — NL→structured params→function):
    ├── analyze            (complex: mode resolution, target inference)
    ├── fit_model           (complex: family selection, param tuning)
    ├── download_tickers    (complex: ticker list from NL description)
    ├── query_data          (complex: NL→SQL translation)
    │
    │ plain tools (1x LLM call — direct arg filling):
    ├── discover_data       (no params)
    ├── discover_analyses   (simple: optional dataset + limit)
    ├── register_dataset    (simple: user provides explicit values)
    ├── list_universe       (simple: optional filter)
    ├── search_tickers      (simple: single query string)
    ├── profile_dataset     (simple: single name)
    ├── forecast_model      (simple: run_id + steps)
    ├── run_diagnostics     (simple: name + column)
    ├── compare_models      (simple: list of run_ids)
    ├── get_run             (simple: single run_id)
    ├── list_runs           (simple: optional dataset + limit)
    ├── list_artifacts      (simple: run_id + optional type)
    └── get_plot            (simple: run_id + name)

Usage::

    from lazybridge.ext.quant_agent import quant_agent

    agent, rt = quant_agent("anthropic")
    resp = agent("Download SPY, AAPL, and MSFT. Analyze their volatility.")
    print(resp.text())
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
- `download_tickers` — describe what data to download (intelligent — extracts tickers from your description)

**Discovery:**
- `discover_data()` — see registered datasets with column roles and quality signals
- `discover_analyses(dataset_name=..., limit=...)` — review completed analyses

**Analysis:**
- `analyze` — describe your analysis goal (intelligent — picks mode, resolves target, tunes params)
  - Supports: recommend, describe, forecast, volatility, regime
- `register_dataset(name, uri, ...)` — register a new data file

**Expert Tools:**
- `fit_model` — describe the model to fit (intelligent — selects family, params)
- `query_data` — describe the SQL query you need (intelligent — translates NL to SQL)
- `forecast_model(run_id, steps)` — generate forecast from a fitted model
- `run_diagnostics(series_name, column)` — stationarity tests on a data column
- `compare_models(run_ids)` — compare multiple model runs
- `profile_dataset(name)` — column-level statistics
- `get_run(run_id)` — retrieve a past model run
- `list_runs(dataset_name=..., limit=...)` — list past model runs
- `list_artifacts(run_id, artifact_type=...)` — list artifacts for a run
- `get_plot(run_id, name)` — get a specific plot

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

    Hybrid architecture:
    - Complex tools (analyze, fit_model, download_tickers, query_data) get
      dedicated sub-agent pipelines via agent_tool() for intelligent parameter
      construction (2x LLM calls but better accuracy).
    - Simple tools (discover_data, get_run, etc.) use direct tool calling
      for efficiency (1x LLM call).

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
        resp = agent("Download SPY and analyze its volatility")
        print(resp.text())
        rt.close()
    """
    from lazybridge import Agent, Tool
    from lazybridge.engines.llm import LLMEngine
    from lazybridge.ext.data_downloader import (
        DataDownloader,
        TickerDatabase,
        build_downloader_skills,
        downloader_skill_tools,
        downloader_tools,
    )
    from lazybridge.ext.stat_runtime.runner import StatRuntime
    from lazybridge.ext.stat_runtime.schemas import (
        AnalyzeInput,
        DownloadTickersInput,
        FitModelInput,
        QueryDataInput,
    )
    from lazybridge.ext.stat_runtime.tools import (
        build_stat_skills,
        stat_skill_tools,
        stat_tools,
    )

    # Runtime
    rt = StatRuntime(artifacts_dir=artifacts_dir)
    effective_cache = cache_dir or str(Path(artifacts_dir) / "ticker_cache")

    # Downloader
    db = TickerDatabase()
    dl = DataDownloader(cache_dir=effective_cache)

    # Get raw tool functions (bound to runtime via closures)
    all_stat = stat_tools(rt, level="all")
    dl_tools_list = downloader_tools(rt, db, dl)

    model_str = model or (provider if isinstance(provider, str) else "anthropic")

    # Build name→Tool lookup
    raw_tools = {t.name: t for t in all_stat + dl_tools_list}

    # Common sub-agent kwargs
    # ---------------------------------------------------------------
    # COMPLEX TOOLS — agent_tool pipelines (NL → structured → function)
    # These benefit from a dedicated LLM step for parameter construction.
    # ---------------------------------------------------------------

    agent_tools: list[Tool] = []

    def _agent_tool(func, *, input_schema, name, description, guidance):
        """Wrap func in an LLM-powered structured-input pipeline (NL → schema → func)."""
        sub = Agent(engine=LLMEngine(model_str, output=input_schema), name=f"{name}_parser")

        async def _invoke(task: str) -> dict:
            env = await sub.run(task)
            if not env.ok:
                return {"error": str(env.error.message)}
            return func(**env.payload.model_dump())

        t = Tool(_invoke, name=name, description=description, guidance=guidance)
        return t

    agent_tools.append(
        _agent_tool(
            raw_tools["analyze"].func,
            input_schema=AnalyzeInput,
            name="analyze",
            description="Run a goal-oriented analysis with automatic model selection. "
            "Describe your analysis goal — mode, target, and params are inferred.",
            guidance="Primary analysis tool. Just describe what you want to analyze and how. "
            "The sub-agent resolves mode, target column, and parameters from your description.",
        )
    )

    agent_tools.append(
        _agent_tool(
            raw_tools["fit_model"].func,
            input_schema=FitModelInput,
            name="fit_model",
            description="Fit a specific statistical model to data. "
            "Describe the model — family, parameters, and data source are inferred.",
            guidance="For custom model fits. Describe the model family (OLS/ARIMA/GARCH/Markov), "
            "target data, and any specific parameters you want.",
        )
    )

    agent_tools.append(
        _agent_tool(
            raw_tools["download_tickers"].func,
            input_schema=DownloadTickersInput,
            name="download_tickers",
            description="Download market data and register for analysis. "
            "Describe what data you need — ticker symbols and date range are extracted.",
            guidance="Describe the data you want. The sub-agent identifies ticker symbols, "
            "date range, and registration preferences from your description.",
        )
    )

    agent_tools.append(
        _agent_tool(
            raw_tools["query_data"].func,
            input_schema=QueryDataInput,
            name="query_data",
            description="Query registered datasets with SQL. "
            "Describe what data you need — the SQL is generated automatically.",
            guidance="Describe the data query in natural language. The sub-agent translates "
            "to SQL using dataset('name') macro. Only SELECT is allowed.",
        )
    )

    # ---------------------------------------------------------------
    # SIMPLE TOOLS — direct tool calling (LLM fills args directly)
    # These have straightforward params the router can handle.
    # ---------------------------------------------------------------

    simple_tool_names = [
        "discover_data",
        "discover_analyses",
        "register_dataset",
        "list_universe",
        "search_tickers",
        "profile_dataset",
        "forecast_model",
        "run_diagnostics",
        "compare_models",
        "get_run",
        "list_runs",
        "list_artifacts",
        "get_plot",
        "list_datasets",
    ]

    simple_tools = [raw_tools[n] for n in simple_tool_names if n in raw_tools]

    # ---------------------------------------------------------------
    # Combine: agent_tools + simple_tools + skills
    # ---------------------------------------------------------------

    all_tools: list[Tool] = agent_tools + simple_tools

    # Skills
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
    _agent_kwargs = {k: v for k, v in agent_kwargs.items() if k not in ("system",)}
    agent = Agent(
        engine=LLMEngine(model_str, system=system or QUANT_SYSTEM_PROMPT),
        name=name,
        description="Quantitative analyst with data download and statistical analysis capabilities",
        tools=all_tools,  # type: ignore[arg-type]
        **_agent_kwargs,
    )

    n_agent = len(agent_tools)
    n_simple = len(simple_tools)
    _logger.info(
        "Created quant_agent: %d agent_tools + %d plain tools (%d tickers in universe)",
        n_agent,
        n_simple,
        len(db.list_all()),
    )

    return agent, rt
