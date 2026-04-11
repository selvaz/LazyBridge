"""LazyTool wrappers for the statistical runtime.

Each tool function catches all exceptions and returns a structured error dict
(never raises to the LLM).  Return values are plain dicts (JSON-serializable).

Usage::

    from lazybridge import LazyAgent
    from lazybridge.stat_runtime.runner import StatRuntime
    from lazybridge.stat_runtime.tools import stat_tools

    rt = StatRuntime()
    agent = LazyAgent("anthropic", tools=stat_tools(rt))
    resp = agent.loop("Register data.parquet, fit GARCH(1,1) on returns")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from lazybridge.lazy_tool import LazyTool
from lazybridge.stat_runtime.schemas import Frequency, ModelFamily

_logger = logging.getLogger(__name__)

# Module-level runtime reference — bound by stat_tools()
_runtime = None


def _get_rt():
    if _runtime is None:
        raise RuntimeError(
            "stat_runtime tools not initialized. Use stat_tools(runtime) first."
        )
    return _runtime


def _error(exc: Exception) -> dict:
    return {"error": True, "type": type(exc).__name__, "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def register_dataset(
    name: Annotated[str, "Logical name for the dataset (e.g. 'equities.daily')"],
    uri: Annotated[str, "File path to a Parquet or CSV file"],
    time_column: Annotated[str | None, "Primary time/date column name"] = None,
    frequency: Annotated[str, "Data frequency: daily, weekly, monthly, quarterly, annual, intraday, irregular"] = "daily",
    entity_keys: Annotated[list[str] | None, "Key columns (e.g. ['symbol', 'country'])"] = None,
) -> dict[str, Any]:
    """Register a dataset file for use in statistical analysis.

    The file is scanned for schema and row count but NOT loaded into memory.
    After registration, use the dataset name in queries and model specs.
    """
    try:
        rt = _get_rt()
        uri_lower = uri.lower()
        if uri_lower.endswith(".csv"):
            meta = rt.catalog.register_csv(
                name, uri, frequency=frequency,
                time_column=time_column, entity_keys=entity_keys or [],
            )
        else:
            meta = rt.catalog.register_parquet(
                name, uri, frequency=frequency,
                time_column=time_column, entity_keys=entity_keys or [],
            )
        return meta.model_dump(mode="json")
    except Exception as exc:
        return _error(exc)


def list_datasets() -> list[dict[str, Any]]:
    """List all registered datasets with their metadata."""
    try:
        rt = _get_rt()
        return [
            {"name": d.name, "uri": d.uri, "format": d.file_format,
             "columns": list(d.columns_schema.keys()), "row_count": d.row_count,
             "time_column": d.time_column, "frequency": str(d.frequency)}
            for d in rt.catalog.list_datasets()
        ]
    except Exception as exc:
        return [_error(exc)]


def profile_dataset(
    name: Annotated[str, "Name of a registered dataset"],
) -> dict[str, Any]:
    """Compute column-level statistics (nulls, min, max, mean, std) for a dataset."""
    try:
        rt = _get_rt()
        result = rt.catalog.profile(name)
        return result.model_dump(mode="json")
    except Exception as exc:
        return _error(exc)


def query_data(
    sql: Annotated[str, "SQL SELECT query. Use dataset('name') to reference registered datasets."],
    max_rows: Annotated[int, "Maximum rows to return"] = 5000,
) -> dict[str, Any]:
    """Execute a SQL query against registered datasets.

    Example: SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY' ORDER BY date
    Only SELECT statements are allowed.  Use the dataset('name') macro.
    """
    try:
        rt = _get_rt()
        result = rt.query_engine.execute(sql, max_rows=max_rows)
        return result.model_dump(mode="json")
    except Exception as exc:
        return _error(exc)


def fit_model(
    family: Annotated[str, "Model family: ols, arima, garch, or markov"],
    target_col: Annotated[str, "Target column name in the dataset"],
    dataset_name: Annotated[str | None, "Registered dataset name"] = None,
    query_sql: Annotated[str | None, "SQL query to extract data (alternative to dataset_name)"] = None,
    exog_cols: Annotated[list[str] | None, "Exogenous/independent variable columns"] = None,
    params: Annotated[dict[str, Any] | None, "Model-specific parameters (e.g. {'p': 1, 'q': 1} for GARCH)"] = None,
    forecast_steps: Annotated[int | None, "Number of forecast steps (None = no forecast)"] = None,
    time_col: Annotated[str | None, "Time column for ordering"] = None,
) -> dict[str, Any]:
    """Fit a statistical model to data.

    Supported families and their key parameters:
      - ols: add_constant (bool, default True)
      - arima: order (tuple, e.g. [1,1,1]), seasonal_order, trend
      - garch: p (int), q (int), vol (str), dist (str), mean (str)
      - markov: k_regimes (int), order (int), switching_variance (bool)

    Returns the run record with metrics, diagnostics, and artifact paths.
    """
    try:
        from lazybridge.stat_runtime.schemas import ModelSpec
        rt = _get_rt()

        spec = ModelSpec(
            family=ModelFamily(family),
            target_col=target_col,
            dataset_name=dataset_name,
            query_sql=query_sql,
            exog_cols=exog_cols or [],
            params=params or {},
            forecast_steps=forecast_steps,
            time_col=time_col,
        )
        run = rt.execute(spec)
        result = run.model_dump(mode="json")
        # Remove datetime objects that might not serialize cleanly
        result["created_at"] = str(result["created_at"])
        return result
    except Exception as exc:
        return _error(exc)


def forecast_model(
    run_id: Annotated[str, "Run ID from a previous fit_model call"],
    steps: Annotated[int, "Number of forecast steps"],
    ci_level: Annotated[float, "Confidence interval level (0-1)"] = 0.95,
) -> dict[str, Any]:
    """Generate a forecast from a previously fitted model."""
    try:
        rt = _get_rt()
        result = rt.forecast(run_id, steps, ci_level=ci_level)
        return result.model_dump(mode="json")
    except Exception as exc:
        return _error(exc)


def run_diagnostics(
    series_name: Annotated[str, "Dataset name for stationarity tests"],
    column: Annotated[str, "Column to test"],
) -> list[dict[str, Any]]:
    """Run stationarity tests (ADF + KPSS) on a data column."""
    try:
        import numpy as np
        from lazybridge.stat_runtime.diagnostics import adf_test, kpss_test
        rt = _get_rt()
        df = rt.catalog.load_df(series_name)
        series = np.array(df[column].to_list(), dtype=float)
        series = series[~np.isnan(series)]
        results = [adf_test(series), kpss_test(series)]
        return [r.model_dump(mode="json") for r in results]
    except Exception as exc:
        return [_error(exc)]


def get_run(
    run_id: Annotated[str, "Run ID to retrieve"],
) -> dict[str, Any]:
    """Retrieve a past model run record with its metrics and artifact paths."""
    try:
        rt = _get_rt()
        run = rt.get_run(run_id)
        if run is None:
            return {"error": True, "message": f"Run '{run_id}' not found"}
        result = run.model_dump(mode="json")
        result["created_at"] = str(result["created_at"])
        return result
    except Exception as exc:
        return _error(exc)


def list_runs(
    dataset_name: Annotated[str | None, "Filter by dataset name"] = None,
    limit: Annotated[int, "Maximum runs to return"] = 20,
) -> list[dict[str, Any]]:
    """List past model runs, optionally filtered by dataset."""
    try:
        rt = _get_rt()
        runs = rt.list_runs(dataset_name=dataset_name, limit=limit)
        return [
            {"run_id": r.run_id, "engine": r.engine, "dataset": r.dataset_name,
             "status": str(r.status), "aic": r.metrics_json.get("aic"),
             "bic": r.metrics_json.get("bic"), "duration": r.duration_secs,
             "created_at": str(r.created_at)}
            for r in runs
        ]
    except Exception as exc:
        return [_error(exc)]


def compare_models(
    run_ids: Annotated[list[str], "List of run IDs to compare"],
) -> dict[str, Any]:
    """Compare multiple model runs by AIC, BIC, and other metrics."""
    try:
        from lazybridge.stat_runtime.diagnostics import compare_models as _compare
        rt = _get_rt()
        runs = [rt.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]
        if not runs:
            return {"error": True, "message": "No valid runs found"}
        result = _compare(runs)
        return result.model_dump(mode="json")
    except Exception as exc:
        return _error(exc)


def list_artifacts(
    run_id: Annotated[str, "Run ID to list artifacts for"],
    artifact_type: Annotated[str | None, "Filter: plot, data, summary, forecast"] = None,
) -> list[dict[str, Any]]:
    """List all artifacts (plots, data, summaries) for a model run."""
    try:
        rt = _get_rt()
        arts = rt.meta_store.list_artifacts(run_id=run_id, artifact_type=artifact_type)
        return [a.model_dump(mode="json") for a in arts]
    except Exception as exc:
        return [_error(exc)]


def get_plot(
    run_id: Annotated[str, "Run ID"],
    name: Annotated[str, "Plot name (e.g. 'residuals', 'volatility', 'forecast', 'regimes')"],
) -> dict[str, Any]:
    """Get the file path for a specific plot from a model run."""
    try:
        rt = _get_rt()
        arts = rt.meta_store.list_artifacts(run_id=run_id, artifact_type="plot")
        for a in arts:
            if a.name == name:
                return {"path": a.path, "name": a.name, "description": a.description}
        available = [a.name for a in arts]
        return {"error": True, "message": f"Plot '{name}' not found. Available: {available}"}
    except Exception as exc:
        return _error(exc)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def stat_tools(runtime) -> list[LazyTool]:
    """Return all stat_runtime tools bound to the given StatRuntime instance.

    Usage::

        from lazybridge.stat_runtime.runner import StatRuntime
        from lazybridge.stat_runtime.tools import stat_tools

        rt = StatRuntime()
        tools = stat_tools(rt)
        agent = LazyAgent("anthropic", tools=tools)
    """
    global _runtime
    _runtime = runtime

    return [
        LazyTool.from_function(
            register_dataset, name="register_dataset",
            description="Register a Parquet or CSV file as a named dataset for analysis",
            guidance="Call this first to make data available. The file is scanned but not loaded.",
        ),
        LazyTool.from_function(
            list_datasets, name="list_datasets",
            description="List all registered datasets with schema and row count",
            guidance="Call to discover what data is available before querying or fitting.",
        ),
        LazyTool.from_function(
            profile_dataset, name="profile_dataset",
            description="Compute column-level statistics for a registered dataset",
            guidance="Call to understand data quality: nulls, ranges, distributions.",
        ),
        LazyTool.from_function(
            query_data, name="query_data",
            description="Run a SQL query on registered datasets using dataset('name') macro",
            guidance="Use SELECT only. Reference datasets with dataset('name'). Example: "
                     "SELECT date, ret FROM dataset('equities') WHERE symbol='SPY' ORDER BY date",
        ),
        LazyTool.from_function(
            fit_model, name="fit_model",
            description="Fit a statistical model (OLS, ARIMA, GARCH, Markov) to data",
            guidance="Specify family and target_col. Key params by family: "
                     "GARCH: p, q; ARIMA: order; Markov: k_regimes. "
                     "Auto-generates diagnostics and plots.",
        ),
        LazyTool.from_function(
            forecast_model, name="forecast_model",
            description="Generate a forecast from a previously fitted model",
            guidance="Provide run_id from fit_model and number of steps to forecast.",
        ),
        LazyTool.from_function(
            run_diagnostics, name="run_diagnostics",
            description="Run stationarity tests (ADF + KPSS) on a data column",
            guidance="Call before fitting to check if data needs differencing.",
        ),
        LazyTool.from_function(
            get_run, name="get_run",
            description="Retrieve a past model run with metrics and artifact paths",
        ),
        LazyTool.from_function(
            list_runs, name="list_runs",
            description="List past model runs, optionally filtered by dataset",
        ),
        LazyTool.from_function(
            compare_models, name="compare_models",
            description="Compare multiple model runs by AIC, BIC, and other metrics",
            guidance="Provide a list of run_ids. Returns a comparison with the best model.",
        ),
        LazyTool.from_function(
            list_artifacts, name="list_artifacts",
            description="List all artifacts (plots, data, summaries) for a model run",
            guidance="Call to see what outputs are available: plots, residuals, forecasts.",
        ),
        LazyTool.from_function(
            get_plot, name="get_plot",
            description="Get the file path for a specific plot from a model run",
            guidance="Common plot names: residuals, acf_pacf, volatility, regimes, forecast.",
        ),
    ]


# ---------------------------------------------------------------------------
# Skill integration
# ---------------------------------------------------------------------------

def build_stat_skills(output_root: str = "./generated_skills") -> dict[str, Any]:
    """Build both skill bundles (tool guide + quant methodology).

    Call once — persists to disk. Returns dict with skill_dir paths.
    """
    from lazybridge.tools.doc_skills import build_skill

    base = Path(__file__).parent / "skill_docs"
    result = {}

    tool_guide_dir = base / "tool_guide"
    if tool_guide_dir.exists() and any(tool_guide_dir.iterdir()):
        result["tool_guide"] = build_skill(
            [str(tool_guide_dir)],
            "stat-tool-guide",
            output_root=output_root,
            description="Comprehensive guide to using the statistical runtime tools. "
                        "Query for parameter details, example workflows, error recovery.",
        )

    quant_dir = base / "quant_methodology"
    if quant_dir.exists() and any(quant_dir.iterdir()):
        result["quant_methodology"] = build_skill(
            [str(quant_dir)],
            "quant-methodology",
            output_root=output_root,
            description="Quantitative finance and econometrics methodology. "
                        "Query for model selection, interpretation, risk analysis.",
        )

    return result


def stat_skill_tools(skill_dir_map: dict[str, Any]) -> list[LazyTool]:
    """Create LazyTool wrappers for the stat skill bundles."""
    from lazybridge.tools.doc_skills import skill_tool

    tools = []
    if "tool_guide" in skill_dir_map:
        tools.append(skill_tool(
            skill_dir_map["tool_guide"]["skill_dir"],
            name="stat_tool_guide",
            guidance="Query this when you need to know HOW to use a statistical tool — "
                     "parameters, formats, workflows, error recovery.",
        ))
    if "quant_methodology" in skill_dir_map:
        tools.append(skill_tool(
            skill_dir_map["quant_methodology"]["skill_dir"],
            name="quant_methodology",
            guidance="Query this when you need to know WHAT to do — model selection, "
                     "interpretation frameworks, risk analysis, reporting standards.",
        ))
    return tools
