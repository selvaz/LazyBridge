"""LazyTool wrappers for the statistical runtime.

Each tool function catches all exceptions and returns a structured error dict
(never raises to the LLM).  Return values are plain dicts (JSON-serializable).

Runtime binding is **closure-based**: each call to ``stat_tools(runtime)``
returns tools permanently bound to that specific runtime instance.  Multiple
runtimes can coexist safely in the same process.

Two-tier architecture::

    # High-level tools (for main agent): discover_data, discover_analyses,
    # analyze, register_dataset
    tools = stat_tools(rt, level="high")

    # Low-level expert tools (for sub-agent): fit_model, query_data, etc.
    tools = stat_tools(rt, level="low")

    # All tools combined
    tools = stat_tools(rt, level="all")   # or just stat_tools(rt)

Usage::

    from lazybridge import LazyAgent
    from lazybridge.stat_runtime.runner import StatRuntime
    from lazybridge.stat_runtime.tools import stat_tools

    rt = StatRuntime()
    agent = LazyAgent("anthropic", tools=stat_tools(rt, level="high"))
    resp = agent.loop("Register data.parquet and analyze the volatility of SPY returns")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from lazybridge.lazy_tool import LazyTool
from lazybridge.stat_runtime.schemas import Frequency, ModelFamily

_logger = logging.getLogger(__name__)


def _error(exc: Exception) -> dict:
    return {"error": True, "type": type(exc).__name__, "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool factory — closure-based, no global state
# ---------------------------------------------------------------------------

def stat_tools(runtime, level: str = "all") -> list[LazyTool]:
    """Return stat_runtime tools bound to the given StatRuntime instance.

    Args:
        runtime: A StatRuntime instance.
        level: Tool tier to return:
            - ``"high"`` — discovery + analyze + register_dataset (4 tools).
              Designed for the main LLM agent.
            - ``"low"`` — all 11 original expert tools (fit_model, query_data,
              etc.). Designed for an expert sub-agent.
            - ``"all"`` — both tiers combined (15 tools). Default.

    Each tool captures ``runtime`` in its closure.  Multiple runtimes can
    coexist: tools from ``stat_tools(rt1)`` will always use ``rt1``, even
    if ``stat_tools(rt2)`` is called later.
    """
    rt = runtime  # captured by all closures below

    # ===================================================================
    # HIGH-LEVEL TOOLS (main agent tier)
    # ===================================================================

    # -- discover_data ---------------------------------------------------

    def discover_data() -> dict[str, Any]:
        """Discover all registered datasets with column types, inferred roles, and suggestions.

        Call this FIRST to understand what data is available before running any analysis.
        Returns column-level type information, inferred roles (target, time, entity key),
        and actionable suggestions for next steps.
        """
        try:
            from lazybridge.stat_runtime.inference import (
                infer_column_roles,
                suggest_for_dataset,
            )
            from lazybridge.stat_runtime.schemas import (
                DataDiscoveryResult,
                DatasetDiscovery,
            )

            datasets_meta = rt.catalog.list_datasets()
            discoveries = []

            for meta in datasets_meta:
                roles = infer_column_roles(meta)
                suggestions = suggest_for_dataset(meta, roles)
                discoveries.append(DatasetDiscovery(
                    name=meta.name,
                    uri=meta.uri,
                    file_format=meta.file_format,
                    frequency=str(meta.frequency),
                    row_count=meta.row_count,
                    time_column=meta.time_column,
                    entity_keys=meta.entity_keys,
                    columns=meta.columns_schema,
                    column_roles=[r.model_dump() for r in roles],
                    suggestions=suggestions,
                    has_profile=bool(meta.profile_json),
                ))

            global_suggestions = []
            if not discoveries:
                global_suggestions.append(
                    "No datasets registered. Call register_dataset() to add data files."
                )

            result = DataDiscoveryResult(
                datasets=discoveries,
                total_datasets=len(discoveries),
                suggestions=global_suggestions,
            )
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- discover_analyses -----------------------------------------------

    def discover_analyses(
        dataset_name: Annotated[str | None, "Filter by dataset name"] = None,
        limit: Annotated[int, "Maximum runs to return"] = 20,
    ) -> dict[str, Any]:
        """Discover completed analysis runs with metrics, diagnostics, and artifact catalogs.

        Call this to see what analyses have already been performed.
        Returns run records enriched with inline metrics, diagnostics pass/fail counts,
        and a complete list of artifacts (plots, data files, summaries) for each run.
        """
        try:
            from lazybridge.stat_runtime.schemas import (
                AnalysisDiscoveryResult,
                ArtifactSummary,
                RunSummary,
            )

            runs = rt.list_runs(dataset_name=dataset_name, limit=limit)
            summaries = []
            best_aic = (None, float("inf"))
            best_bic = (None, float("inf"))
            ds_set: set[str] = set()

            for run in runs:
                # Extract metrics
                m = run.metrics_json or {}
                aic = m.get("aic")
                bic = m.get("bic")
                ll = m.get("log_likelihood")

                if aic is not None and aic < best_aic[1]:
                    best_aic = (run.run_id, aic)
                if bic is not None and bic < best_bic[1]:
                    best_bic = (run.run_id, bic)

                # Extract diagnostics summary
                diags = run.diagnostics_json or []
                d_passed = sum(1 for d in diags if d.get("passed") is True)
                d_failed = sum(1 for d in diags if d.get("passed") is False)

                # Extract spec highlights
                spec = run.spec_json or {}
                target = spec.get("target_col", "")
                model_params = spec.get("params", {})

                # Build artifact catalog
                artifacts = []
                try:
                    arts = rt.meta_store.list_artifacts(run_id=run.run_id)
                    for a in arts:
                        artifacts.append(ArtifactSummary(
                            name=a.name,
                            artifact_type=a.artifact_type,
                            file_format=a.file_format,
                            path=a.path,
                            description=a.description,
                        ))
                except Exception:
                    pass  # artifact listing failure should not break discovery

                if run.dataset_name:
                    ds_set.add(run.dataset_name)

                summaries.append(RunSummary(
                    run_id=run.run_id,
                    dataset_name=run.dataset_name,
                    engine=run.engine,
                    status=str(run.status),
                    created_at=str(run.created_at),
                    duration_secs=run.duration_secs,
                    aic=aic,
                    bic=bic,
                    log_likelihood=ll,
                    diagnostics_passed=d_passed,
                    diagnostics_failed=d_failed,
                    diagnostics_total=d_passed + d_failed,
                    target_col=target,
                    model_params=model_params,
                    artifacts=artifacts,
                    error_message=run.error_message,
                ))

            # Suggestions
            suggestions = []
            failed = [s for s in summaries if s.status == "failed"]
            if failed:
                suggestions.append(
                    f"{len(failed)} run(s) failed. Check error_message for details."
                )
            engines_used = {s.engine for s in summaries if s.engine}
            if len(engines_used) >= 2:
                suggestions.append(
                    "Multiple model families fitted. Use compare_models() to evaluate."
                )

            result = AnalysisDiscoveryResult(
                runs=summaries,
                total_runs=len(summaries),
                datasets_analyzed=sorted(ds_set),
                best_by_aic=best_aic[0],
                best_by_bic=best_bic[0],
                suggestions=suggestions,
            )
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- analyze ---------------------------------------------------------

    def analyze(
        family: Annotated[str, "Model family: ols, arima, garch, or markov"],
        target_col: Annotated[str, "Target column name"],
        dataset_name: Annotated[str | None, "Registered dataset name"] = None,
        query_sql: Annotated[str | None, "SQL query using dataset('name') macro"] = None,
        exog_cols: Annotated[list[str] | None, "Exogenous variable columns"] = None,
        params: Annotated[dict[str, Any] | None, "Model parameters (e.g. {'p': 1, 'q': 1})"] = None,
        forecast_steps: Annotated[int | None, "Forecast steps (None = no forecast)"] = None,
        time_col: Annotated[str | None, "Time column for ordering"] = None,
    ) -> dict[str, Any]:
        """Run a complete statistical analysis: fit model, diagnostics, plots, and interpretation.

        This is the primary analysis tool. It calls fit_model internally but returns a richer
        result with inline artifact catalog, diagnostic health assessment, interpretation hints,
        and suggested next steps. Prefer this over fit_model for new analyses.
        """
        try:
            from lazybridge.stat_runtime.inference import build_interpretation
            from lazybridge.stat_runtime.schemas import (
                AnalysisResult,
                ArtifactSummary,
                ModelSpec,
            )

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

            # Build artifact catalog
            plots = []
            data_arts = []
            try:
                arts = rt.meta_store.list_artifacts(run_id=run.run_id)
                for a in arts:
                    summary = ArtifactSummary(
                        name=a.name,
                        artifact_type=a.artifact_type,
                        file_format=a.file_format,
                        path=a.path,
                        description=a.description,
                    )
                    if a.artifact_type == "plot":
                        plots.append(summary)
                    else:
                        data_arts.append(summary)
            except Exception:
                pass

            # Diagnostics assessment
            diags = run.diagnostics_json or []
            d_passed = sum(1 for d in diags if d.get("passed") is True)
            d_failed = sum(1 for d in diags if d.get("passed") is False)
            # Model adequate if no Ljung-Box failures (Jarque-Bera is informational)
            lb_failures = [
                d for d in diags
                if d.get("passed") is False
                and "ljung" in d.get("test_name", "").lower()
            ]
            model_adequate = run.status == "success" and len(lb_failures) == 0

            # Forecast data
            forecast_data = None
            if forecast_steps and run.status == "success":
                try:
                    fc_arts = rt.meta_store.list_artifacts(
                        run_id=run.run_id, artifact_type="forecast",
                    )
                    if fc_arts:
                        fc_json = rt.artifacts.read_json(
                            run.run_id, "forecast", artifact_type="data",
                        )
                        forecast_data = fc_json
                except Exception:
                    pass

            # Interpretation
            interp, warns, nexts = build_interpretation(run, family)

            result = AnalysisResult(
                run_id=run.run_id,
                status=str(run.status),
                engine=run.engine,
                dataset_name=run.dataset_name,
                target_col=target_col,
                params=run.params_json or {},
                metrics=run.metrics_json or {},
                fit_summary=run.fit_summary or "",
                diagnostics=diags,
                diagnostics_passed=d_passed,
                diagnostics_failed=d_failed,
                model_adequate=model_adequate,
                forecast=forecast_data,
                plots=plots,
                data_artifacts=data_arts,
                interpretation=interp,
                warnings=warns,
                next_steps=nexts,
                duration_secs=run.duration_secs,
                error_message=run.error_message,
            )
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # ===================================================================
    # LOW-LEVEL EXPERT TOOLS
    # ===================================================================

    # -- register_dataset ------------------------------------------------

    def register_dataset(
        name: Annotated[str, "Logical name for the dataset (e.g. 'equities.daily')"],
        uri: Annotated[str, "File path to a Parquet or CSV file"],
        time_column: Annotated[str | None, "Primary time/date column name"] = None,
        frequency: Annotated[str, "Data frequency: daily, weekly, monthly, quarterly, annual, intraday, irregular"] = "daily",
        entity_keys: Annotated[list[str] | None, "Key columns (e.g. ['symbol', 'country'])"] = None,
    ) -> dict[str, Any]:
        """Register a dataset file for use in statistical analysis."""
        try:
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

    # -- list_datasets ---------------------------------------------------

    def list_datasets() -> list[dict[str, Any]]:
        """List all registered datasets with their metadata."""
        try:
            return [
                {"name": d.name, "uri": d.uri, "format": d.file_format,
                 "columns": list(d.columns_schema.keys()), "row_count": d.row_count,
                 "time_column": d.time_column, "frequency": str(d.frequency)}
                for d in rt.catalog.list_datasets()
            ]
        except Exception as exc:
            return [_error(exc)]

    # -- profile_dataset -------------------------------------------------

    def profile_dataset(
        name: Annotated[str, "Name of a registered dataset"],
    ) -> dict[str, Any]:
        """Compute column-level statistics (nulls, min, max, mean, std) for a dataset."""
        try:
            result = rt.catalog.profile(name)
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- query_data ------------------------------------------------------

    def query_data(
        sql: Annotated[str, "SQL SELECT query. Use dataset('name') to reference registered datasets."],
        max_rows: Annotated[int, "Maximum rows to return"] = 5000,
    ) -> dict[str, Any]:
        """Execute a SQL query against registered datasets.

        Example: SELECT date, ret FROM dataset('equities') WHERE symbol = 'SPY' ORDER BY date
        Only SELECT statements are allowed.  Use the dataset('name') macro.
        Direct file access (read_parquet, etc.) is blocked — use dataset('name') instead.
        """
        try:
            result = rt.query_engine.execute(sql, max_rows=max_rows)
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- fit_model -------------------------------------------------------

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
            result["created_at"] = str(result["created_at"])
            return result
        except Exception as exc:
            return _error(exc)

    # -- forecast_model --------------------------------------------------

    def forecast_model(
        run_id: Annotated[str, "Run ID from a previous fit_model call"],
        steps: Annotated[int, "Number of forecast steps"],
        ci_level: Annotated[float, "Confidence interval level (0-1)"] = 0.95,
    ) -> dict[str, Any]:
        """Generate a forecast from a previously fitted model."""
        try:
            result = rt.forecast(run_id, steps, ci_level=ci_level)
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- run_diagnostics -------------------------------------------------

    def run_diagnostics(
        series_name: Annotated[str, "Dataset name for stationarity tests"],
        column: Annotated[str, "Column to test"],
    ) -> list[dict[str, Any]]:
        """Run stationarity tests (ADF + KPSS) on a data column."""
        try:
            import numpy as np
            from lazybridge.stat_runtime.diagnostics import adf_test, kpss_test
            df = rt.catalog.load_df(series_name)
            series = np.array(df[column].to_list(), dtype=float)
            series = series[~np.isnan(series)]
            results = [adf_test(series), kpss_test(series)]
            return [r.model_dump(mode="json") for r in results]
        except Exception as exc:
            return [_error(exc)]

    # -- get_run ---------------------------------------------------------

    def get_run(
        run_id: Annotated[str, "Run ID to retrieve"],
    ) -> dict[str, Any]:
        """Retrieve a past model run record with its metrics and artifact paths."""
        try:
            run = rt.get_run(run_id)
            if run is None:
                return {"error": True, "message": f"Run '{run_id}' not found"}
            result = run.model_dump(mode="json")
            result["created_at"] = str(result["created_at"])
            return result
        except Exception as exc:
            return _error(exc)

    # -- list_runs -------------------------------------------------------

    def list_runs(
        dataset_name: Annotated[str | None, "Filter by dataset name"] = None,
        limit: Annotated[int, "Maximum runs to return"] = 20,
    ) -> list[dict[str, Any]]:
        """List past model runs, optionally filtered by dataset."""
        try:
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

    # -- compare_models --------------------------------------------------

    def compare_models(
        run_ids: Annotated[list[str], "List of run IDs to compare"],
    ) -> dict[str, Any]:
        """Compare multiple model runs by AIC, BIC, and other metrics."""
        try:
            from lazybridge.stat_runtime.diagnostics import compare_models as _compare
            runs = [rt.get_run(rid) for rid in run_ids]
            runs = [r for r in runs if r is not None]
            if not runs:
                return {"error": True, "message": "No valid runs found"}
            result = _compare(runs)
            return result.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- list_artifacts --------------------------------------------------

    def list_artifacts(
        run_id: Annotated[str, "Run ID to list artifacts for"],
        artifact_type: Annotated[str | None, "Filter: plot, data, summary, forecast"] = None,
    ) -> list[dict[str, Any]]:
        """List all artifacts (plots, data, summaries) for a model run."""
        try:
            arts = rt.meta_store.list_artifacts(run_id=run_id, artifact_type=artifact_type)
            return [a.model_dump(mode="json") for a in arts]
        except Exception as exc:
            return [_error(exc)]

    # -- get_plot --------------------------------------------------------

    def get_plot(
        run_id: Annotated[str, "Run ID"],
        name: Annotated[str, "Plot name (e.g. 'residuals', 'volatility', 'forecast', 'regimes')"],
    ) -> dict[str, Any]:
        """Get the file path for a specific plot from a model run."""
        try:
            arts = rt.meta_store.list_artifacts(run_id=run_id, artifact_type="plot")
            for a in arts:
                if a.name == name:
                    return {"path": a.path, "name": a.name, "description": a.description}
            available = [a.name for a in arts]
            return {"error": True, "message": f"Plot '{name}' not found. Available: {available}"}
        except Exception as exc:
            return _error(exc)

    # -- Assemble tools by tier ------------------------------------------

    high_level_tools = [
        LazyTool.from_function(
            discover_data, name="discover_data",
            description="Discover all registered datasets with column roles, types, and suggestions",
            guidance="Call this first to understand what data is available. "
                     "Returns enriched metadata with inferred column roles and actionable suggestions.",
        ),
        LazyTool.from_function(
            discover_analyses, name="discover_analyses",
            description="Discover completed analysis runs with metrics, diagnostics, and artifact catalogs",
            guidance="Call to see what analyses and artifacts already exist. "
                     "Shows metrics, diagnostics health, and all plots/data per run.",
        ),
        LazyTool.from_function(
            analyze, name="analyze",
            description="Run a complete analysis: fit model, diagnostics, plots, and interpretation",
            guidance="Primary analysis tool. Returns enriched output with interpretation, "
                     "diagnostics health, artifact catalog, and suggested next steps. "
                     "Prefer this over fit_model for new analyses.",
        ),
        LazyTool.from_function(
            register_dataset, name="register_dataset",
            description="Register a Parquet or CSV file as a named dataset for analysis",
            guidance="Call to make a new data file available. The file is scanned but not loaded. "
                     "After registering, call discover_data() to see the enriched view.",
        ),
    ]

    low_level_tools = [
        LazyTool.from_function(
            list_datasets, name="list_datasets",
            description="List all registered datasets with schema and row count",
            guidance="Expert tool. Prefer discover_data() for a richer view with column roles. "
                     "Use this for a quick name/schema check.",
        ),
        LazyTool.from_function(
            profile_dataset, name="profile_dataset",
            description="Compute column-level statistics for a registered dataset",
            guidance="Call to understand data quality: nulls, ranges, distributions.",
        ),
        LazyTool.from_function(
            query_data, name="query_data",
            description="Run a SQL query on registered datasets using dataset('name') macro",
            guidance="Use SELECT only. Reference datasets with dataset('name'). "
                     "Direct file access (read_parquet etc.) is blocked. "
                     "Example: SELECT date, ret FROM dataset('equities') WHERE symbol='SPY' ORDER BY date",
        ),
        LazyTool.from_function(
            fit_model, name="fit_model",
            description="Fit a statistical model (OLS, ARIMA, GARCH, Markov) to data",
            guidance="Expert tool. Prefer analyze() for standard analyses — it returns richer output. "
                     "Use fit_model for programmatic fits or when you need the raw RunRecord. "
                     "Key params by family: GARCH: p, q; ARIMA: order; Markov: k_regimes.",
        ),
        LazyTool.from_function(
            forecast_model, name="forecast_model",
            description="Generate a forecast from a previously fitted model",
            guidance="Provide run_id from fit_model/analyze and number of steps to forecast.",
        ),
        LazyTool.from_function(
            run_diagnostics, name="run_diagnostics",
            description="Run stationarity tests (ADF + KPSS) on a data column",
            guidance="Call before fitting to check if data needs differencing.",
        ),
        LazyTool.from_function(
            get_run, name="get_run",
            description="Retrieve a past model run with full metrics and artifact paths",
        ),
        LazyTool.from_function(
            list_runs, name="list_runs",
            description="List past model runs, optionally filtered by dataset",
            guidance="Expert tool. Prefer discover_analyses() for enriched view "
                     "with inline metrics and artifact catalogs.",
        ),
        LazyTool.from_function(
            compare_models, name="compare_models",
            description="Compare multiple model runs by AIC, BIC, and other metrics",
            guidance="Provide a list of run_ids. Returns a comparison with the best model.",
        ),
        LazyTool.from_function(
            list_artifacts, name="list_artifacts",
            description="List all artifacts (plots, data, summaries) for a model run",
            guidance="Expert tool. discover_analyses() includes artifacts inline. "
                     "Use this for type-filtered artifact lookups.",
        ),
        LazyTool.from_function(
            get_plot, name="get_plot",
            description="Get the file path for a specific plot from a model run",
            guidance="Common plot names: residuals, acf_pacf, volatility, regimes, forecast.",
        ),
    ]

    if level == "high":
        return high_level_tools
    if level == "low":
        return low_level_tools
    return high_level_tools + low_level_tools


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


# ---------------------------------------------------------------------------
# Pre-configured agent with two-tier architecture
# ---------------------------------------------------------------------------

def stat_agent(
    provider: str = "anthropic",
    *,
    model: str | None = None,
    db: str | None = None,
    artifacts_dir: str = "artifacts",
    include_skills: bool = True,
    expert_mode: bool = True,
    name: str = "stat_analyst",
    system: str | None = None,
    **agent_kwargs: Any,
) -> tuple[Any, Any]:
    """Create a pre-configured LazyAgent with stat_runtime tools.

    Two operating modes:

    **expert_mode=True** (default):
        Main agent gets high-level tools (discover_data, discover_analyses,
        analyze, register_dataset) plus a ``delegate_to_expert`` tool that
        forwards tasks to an inner agent with all low-level tools.

    **expert_mode=False**:
        Main agent gets all tools in a flat list (no delegation).

    Returns:
        (agent, runtime) tuple. The runtime is returned so the caller can
        register datasets programmatically or close it when done.

    Usage::

        agent, rt = stat_agent("anthropic")
        agent.loop("Register data.parquet and analyze volatility of SPY returns")
        rt.close()
    """
    from lazybridge.lazy_agent import LazyAgent
    from lazybridge.stat_runtime.runner import StatRuntime

    rt = StatRuntime(db=db, artifacts_dir=artifacts_dir)

    default_system = (
        "You are a quantitative analyst. Follow this workflow:\n"
        "1. discover_data() — understand available datasets\n"
        "2. discover_analyses() — see what's been done\n"
        "3. analyze() — run new analyses\n"
        "4. compare_models() — compare runs\n\n"
        "Always check diagnostics before trusting results. "
        "Report artifact paths so the user can view plots."
    )

    all_tools: list[LazyTool] = []

    if expert_mode:
        # Main agent: high-level tools
        all_tools.extend(stat_tools(rt, level="high"))

        # Inner expert agent with low-level tools
        expert_tools = stat_tools(rt, level="low")
        if include_skills:
            try:
                skill_dirs = build_stat_skills()
                expert_tools.extend(stat_skill_tools(skill_dirs))
            except Exception:
                _logger.warning("Failed to build stat skills for expert agent", exc_info=True)

        expert = LazyAgent(
            provider,
            model=model,
            name="stat_expert",
            description="Expert statistical analyst with low-level tool access",
            system="You are an expert statistical analyst. Use the available tools "
                   "to fulfill the task precisely. Return results as structured data.",
            tools=expert_tools,
            **agent_kwargs,
        )

        # Wrap expert as a tool for the main agent
        expert_tool = expert.as_tool(
            name="delegate_to_expert",
            description="Delegate a task to the expert statistical agent for fine-grained control",
            guidance="Use when the user needs specific model parameters, custom SQL queries, "
                     "individual diagnostic tests, manual plot retrieval, or model comparison. "
                     "The expert has access to: fit_model, forecast_model, query_data, "
                     "profile_dataset, run_diagnostics, compare_models, get_run, list_runs, "
                     "list_datasets, list_artifacts, get_plot.",
        )
        all_tools.append(expert_tool)

        # Add skills to main agent too
        if include_skills:
            try:
                skill_dirs = build_stat_skills()
                all_tools.extend(stat_skill_tools(skill_dirs))
            except Exception:
                pass

    else:
        # Flat mode: all tools
        all_tools.extend(stat_tools(rt, level="all"))
        if include_skills:
            try:
                skill_dirs = build_stat_skills()
                all_tools.extend(stat_skill_tools(skill_dirs))
            except Exception:
                _logger.warning("Failed to build stat skills", exc_info=True)

    if expert_mode:
        default_system += (
            "\n\nUse delegate_to_expert() when the user needs specific model parameters, "
            "custom SQL queries, or fine-grained control over the analysis pipeline."
        )

    agent = LazyAgent(
        provider,
        model=model,
        name=name,
        description="Statistical analyst with quantitative analysis capabilities",
        system=system or default_system,
        tools=all_tools,
        **agent_kwargs,
    )

    return agent, rt
