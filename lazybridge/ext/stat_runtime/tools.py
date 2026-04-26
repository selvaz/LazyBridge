"""Tool wrappers for the statistical runtime.

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

    from lazybridge import Agent
    from lazybridge.ext.stat_runtime.runner import StatRuntime
    from lazybridge.ext.stat_runtime.tools import stat_tools

    rt = StatRuntime()
    agent = Agent("anthropic", tools=stat_tools(rt, level="high"))
    resp = agent("Register data.parquet and analyze the volatility of SPY returns")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from lazybridge import Tool
from lazybridge.ext.stat_runtime.schemas import ModelFamily

_logger = logging.getLogger(__name__)


def _error(exc: Exception) -> dict:
    return {"error": True, "type": type(exc).__name__, "message": str(exc)}


# ---------------------------------------------------------------------------
# Tool factory — closure-based, no global state
# ---------------------------------------------------------------------------


def stat_tools(runtime, level: str = "all") -> list[Tool]:
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
        quality signals (missingness, cardinality from cached profile), a natural language
        summary, and actionable suggestions for next steps.
        """
        try:
            from lazybridge.ext.stat_runtime.inference import (
                generate_dataset_summary,
                infer_column_roles,
                suggest_for_dataset,
            )
            from lazybridge.ext.stat_runtime.schemas import (
                ColumnSignals,
                DataDiscoveryResult,
                DatasetDiscovery,
            )

            datasets_meta = rt.catalog.list_datasets()
            discoveries = []

            for meta in datasets_meta:
                roles = infer_column_roles(meta)
                suggestions = suggest_for_dataset(meta, roles)
                summary = generate_dataset_summary(meta, roles)

                # Extract column signals from cached profile
                col_signals: dict[str, ColumnSignals] = {}
                if meta.profile_json and "columns" in meta.profile_json:
                    for col_name, col_data in meta.profile_json["columns"].items():
                        if isinstance(col_data, dict):
                            col_signals[col_name] = ColumnSignals(
                                null_pct=col_data.get("null_pct"),
                                unique_count=col_data.get("unique_count"),
                                min_val=col_data.get("min_val"),
                                max_val=col_data.get("max_val"),
                                mean=col_data.get("mean"),
                            )

                discoveries.append(
                    DatasetDiscovery(
                        name=meta.name,
                        uri=meta.uri,
                        file_format=meta.file_format,
                        frequency=str(meta.frequency),
                        row_count=meta.row_count,
                        time_column=meta.time_column,
                        entity_keys=meta.entity_keys,
                        columns=meta.columns_schema,
                        column_roles=[r.model_dump() for r in roles],  # type: ignore[misc]
                        column_signals=col_signals,
                        suggestions=suggestions,
                        has_profile=bool(meta.profile_json),
                        business_description=meta.business_description,
                        canonical_target=meta.canonical_target,
                        summary=summary,
                    )
                )

            global_suggestions = []
            if not discoveries:
                global_suggestions.append("No datasets registered. Call register_dataset() to add data files.")

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
            from lazybridge.ext.stat_runtime.schemas import (
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
                        artifacts.append(
                            ArtifactSummary(
                                name=a.name,
                                artifact_type=a.artifact_type,
                                file_format=a.file_format,
                                path=a.path,
                                description=a.description,
                            )
                        )
                except Exception:
                    pass  # artifact listing failure should not break discovery

                if run.dataset_name:
                    ds_set.add(run.dataset_name)

                summaries.append(
                    RunSummary(
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
                    )
                )

            # Suggestions
            suggestions = []
            failed = [s for s in summaries if s.status == "failed"]
            if failed:
                suggestions.append(f"{len(failed)} run(s) failed. Check error_message for details.")
            engines_used = {s.engine for s in summaries if s.engine}
            if len(engines_used) >= 2:
                suggestions.append("Multiple model families fitted. Use compare_models() to evaluate.")

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
        dataset_name: Annotated[str, "Registered dataset name"],
        target_col: Annotated[str | None, "Target column (auto-resolved from metadata/inference if omitted)"] = None,
        mode: Annotated[
            str,
            "Analysis goal: describe, forecast, volatility, regime, recommend — or explicit family: ols, arima, garch, markov",
        ] = "recommend",
        time_col: Annotated[str | None, "Time column for ordering (auto-detected if not set)"] = None,
        forecast_steps: Annotated[int | None, "Forecast steps (None = auto based on mode)"] = None,
        group_col: Annotated[str | None, "Column to filter/segment by (e.g. 'symbol')"] = None,
        group_value: Annotated[str | None, "Value to filter group_col to (e.g. 'SPY')"] = None,
        params: Annotated[dict[str, Any] | None, "Expert override: model parameters"] = None,
    ) -> dict[str, Any]:
        """Run a goal-oriented statistical analysis with automatic model selection.

        The primary analysis tool. Pick a MODE (your goal) instead of a model family:
          - describe: data profiling, stationarity tests, distribution analysis
          - forecast: time-series forecasting (auto-selects ARIMA)
          - volatility: volatility modeling (auto-selects GARCH)
          - regime: regime detection (auto-selects Markov switching)
          - recommend: inspects the data and picks the best analysis automatically

        target_col is optional for recommend and describe modes — the runtime will
        auto-resolve from canonical_target metadata or column role inference.

        Returns enriched output with: why the analysis was chosen, model assumptions,
        interpretation, diagnostics health, artifact catalog, and suggested next steps.
        """
        try:
            from lazybridge.ext.stat_runtime.inference import (
                build_interpretation,
                get_model_assumptions,
                infer_column_roles,
                resolve_analysis_mode,
            )
            from lazybridge.ext.stat_runtime.schemas import (
                AnalysisResult,
                ArtifactSummary,
                ModelSpec,
            )

            # Load dataset metadata
            meta = rt.catalog.get(dataset_name)
            if meta is None:
                return _error(ValueError(f"Dataset '{dataset_name}' is not registered. Call register_dataset() first."))
            roles = infer_column_roles(meta)

            # --- Target resolution ---
            target_resolution_reason = ""
            resolved_target = target_col

            if not resolved_target:
                # 1. Try canonical_target from metadata
                if meta.canonical_target and meta.canonical_target in meta.columns_schema:
                    resolved_target = meta.canonical_target
                    target_resolution_reason = (
                        f"Auto-selected target '{resolved_target}' from dataset canonical_target metadata."
                    )
                else:
                    # 2. Try inferred target candidates
                    candidates = [r.column for r in roles if r.inferred_role == "target"]
                    if len(candidates) == 1:
                        resolved_target = candidates[0]
                        target_resolution_reason = (
                            f"Auto-selected target '{resolved_target}' — only inferred target candidate."
                        )
                    elif len(candidates) > 1:
                        # Ambiguous — return structured response
                        return {
                            "error": True,
                            "type": "AmbiguousTarget",
                            "message": (
                                f"Multiple target candidates found: {candidates}. "
                                f"Please specify target_col explicitly, or set "
                                f"canonical_target when registering the dataset."
                            ),
                            "candidates": candidates,
                        }
                    else:
                        # No candidates — require explicit for non-describe modes
                        if mode.lower() not in ("describe",):
                            return {
                                "error": True,
                                "type": "MissingTarget",
                                "message": (
                                    "No target column specified and none could be inferred. "
                                    "Please provide target_col or set canonical_target "
                                    "when registering the dataset."
                                ),
                                "available_columns": list(meta.columns_schema.keys()),
                            }
                        # For describe, pick the first numeric column as a reasonable default
                        for r in roles:
                            if r.inferred_role in ("target", "exogenous"):
                                resolved_target = r.column
                                target_resolution_reason = (
                                    f"Describe mode: auto-selected '{resolved_target}' "
                                    f"as the first numeric column for profiling."
                                )
                                break
                        if not resolved_target:
                            return _error(ValueError("No numeric columns found for describe mode."))
            else:
                target_resolution_reason = f"Explicit target_col='{resolved_target}'."

            # --- Group filtering validation (safe, no SQL injection) ---
            query_sql = None
            effective_dataset = dataset_name

            if group_col or group_value:
                if not group_col or not group_value:
                    return _error(ValueError("Both group_col and group_value must be provided together."))
                # Validate group_col is a real column
                if group_col not in meta.columns_schema:
                    return _error(
                        ValueError(
                            f"group_col '{group_col}' not found in dataset schema. "
                            f"Available columns: {list(meta.columns_schema.keys())}"
                        )
                    )
                # Validate group_col is a safe identifier (alphanumeric + underscore only)
                import re as _re

                if not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", group_col):
                    return _error(
                        ValueError(
                            f"group_col '{group_col}' contains invalid characters. "
                            f"Only alphanumeric characters and underscores are allowed."
                        )
                    )
                # Sanitize group_value: escape single quotes
                safe_value = group_value.replace("'", "''")
                # Build safe SQL with validated column and escaped value
                order_clause = f" ORDER BY {time_col}" if time_col else ""
                query_sql = (
                    f"SELECT * FROM dataset('{dataset_name}') WHERE \"{group_col}\" = '{safe_value}'{order_clause}"
                )
                effective_dataset = None  # type: ignore[assignment]

            # --- Auto-detect time_col ---
            if not time_col and meta.time_column:
                time_col = meta.time_column

            # --- Resolve mode ---
            family, rationale, assumptions = resolve_analysis_mode(
                mode,
                meta,
                roles,
                resolved_target,
            )
            if not assumptions:
                assumptions = get_model_assumptions(family)

            # Auto forecast steps for forecast/volatility modes
            if forecast_steps is None and mode.lower() in ("forecast", "volatility"):
                forecast_steps = 20

            spec = ModelSpec(
                family=ModelFamily(family),
                target_col=resolved_target,
                dataset_name=effective_dataset,
                query_sql=query_sql,
                exog_cols=[],
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
                    art_summary = ArtifactSummary(
                        name=a.name,
                        artifact_type=a.artifact_type,
                        file_format=a.file_format,
                        path=a.path,
                        description=a.description,
                    )
                    if a.artifact_type == "plot":
                        plots.append(art_summary)
                    else:
                        data_arts.append(art_summary)
            except Exception:
                pass

            # Diagnostics assessment
            diags = run.diagnostics_json or []
            d_passed = sum(1 for d in diags if d.get("passed") is True)
            d_failed = sum(1 for d in diags if d.get("passed") is False)
            lb_failures = [d for d in diags if d.get("passed") is False and "ljung" in d.get("test_name", "").lower()]
            model_adequate = run.status == "success" and len(lb_failures) == 0

            # Forecast data
            forecast_data = None
            if forecast_steps and run.status == "success":
                try:
                    fc_arts = rt.meta_store.list_artifacts(
                        run_id=run.run_id,
                        artifact_type="forecast",
                    )
                    if fc_arts:
                        fc_json = rt.artifacts.read_json(
                            run.run_id,
                            "forecast",
                            artifact_type="data",
                        )
                        forecast_data = fc_json
                except Exception:
                    pass

            # Interpretation
            interp, warns, nexts = build_interpretation(run, family)

            # Add target resolution info
            if target_resolution_reason and target_col is None:
                interp.insert(0, target_resolution_reason)

            result = AnalysisResult(
                run_id=run.run_id,
                status=str(run.status),
                engine=run.engine,
                dataset_name=run.dataset_name or dataset_name,
                target_col=resolved_target,
                mode=mode,
                mode_rationale=rationale,
                assumptions=assumptions,
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
        frequency: Annotated[
            str, "Data frequency: daily, weekly, monthly, quarterly, annual, intraday, irregular"
        ] = "daily",
        entity_keys: Annotated[list[str] | None, "Key columns (e.g. ['symbol', 'country'])"] = None,
        business_description: Annotated[
            str | None, "What this dataset represents (e.g. 'Daily S&P 500 returns')"
        ] = None,
        canonical_target: Annotated[str | None, "Preferred target column for analysis (must exist in the data)"] = None,
        identifiers_to_ignore: Annotated[list[str] | None, "Columns to exclude from modeling (IDs, hashes)"] = None,
    ) -> dict[str, Any]:
        """Register a dataset file for use in statistical analysis.

        Optionally provide semantic metadata to improve analysis quality:
        - business_description: what the data represents
        - canonical_target: preferred target column (used by analyze mode=recommend)
        - identifiers_to_ignore: columns that should not be used as features
        """
        try:
            kwargs = dict(
                frequency=frequency,
                time_column=time_column,
                entity_keys=entity_keys or [],
                business_description=business_description,
                canonical_target=canonical_target,
                identifiers_to_ignore=identifiers_to_ignore or [],
            )
            uri_lower = uri.lower()
            if uri_lower.endswith(".csv"):
                meta = rt.catalog.register_csv(name, uri, **kwargs)
            else:
                meta = rt.catalog.register_parquet(name, uri, **kwargs)
            return meta.model_dump(mode="json")
        except Exception as exc:
            return _error(exc)

    # -- list_datasets ---------------------------------------------------

    def list_datasets() -> list[dict[str, Any]]:
        """List all registered datasets with their metadata."""
        try:
            return [
                {
                    "name": d.name,
                    "uri": d.uri,
                    "format": d.file_format,
                    "columns": list(d.columns_schema.keys()),
                    "row_count": d.row_count,
                    "time_column": d.time_column,
                    "frequency": str(d.frequency),
                }
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
            result = rt.query_engine.execute(sql, max_rows=min(max_rows, 50000))
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
            from lazybridge.ext.stat_runtime.schemas import ModelSpec

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

            from lazybridge.ext.stat_runtime.diagnostics import adf_test, kpss_test

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
                {
                    "run_id": r.run_id,
                    "engine": r.engine,
                    "dataset": r.dataset_name,
                    "status": str(r.status),
                    "aic": r.metrics_json.get("aic"),
                    "bic": r.metrics_json.get("bic"),
                    "duration": r.duration_secs,
                    "created_at": str(r.created_at),
                }
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
            from lazybridge.ext.stat_runtime.diagnostics import compare_models as _compare

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
        Tool(
            discover_data,
            name="discover_data",
            description="Discover all registered datasets with column roles, types, and suggestions",
            guidance="Call this first to understand what data is available. "
            "Returns enriched metadata with inferred column roles and actionable suggestions.",
        ),
        Tool(
            discover_analyses,
            name="discover_analyses",
            description="Discover completed analysis runs with metrics, diagnostics, and artifact catalogs",
            guidance="Call to see what analyses and artifacts already exist. "
            "Shows metrics, diagnostics health, and all plots/data per run.",
        ),
        Tool(
            analyze,
            name="analyze",
            description="Run a goal-oriented analysis with automatic model selection",
            guidance="Primary analysis tool. Use mode='recommend' (default) to let the "
            "runtime pick the best analysis, or specify a goal: describe, forecast, "
            "volatility, regime. Returns interpretation, assumptions, diagnostics, "
            "artifact catalog, and suggested next steps.",
        ),
        Tool(
            register_dataset,
            name="register_dataset",
            description="Register a Parquet or CSV file as a named dataset for analysis",
            guidance="Call to make a new data file available. The file is scanned but not loaded. "
            "After registering, call discover_data() to see the enriched view.",
        ),
    ]

    low_level_tools = [
        Tool(
            list_datasets,
            name="list_datasets",
            description="List all registered datasets with schema and row count",
            guidance="Expert tool. Prefer discover_data() for a richer view with column roles. "
            "Use this for a quick name/schema check.",
        ),
        Tool(
            profile_dataset,
            name="profile_dataset",
            description="Compute column-level statistics for a registered dataset",
            guidance="Call to understand data quality: nulls, ranges, distributions.",
        ),
        Tool(
            query_data,
            name="query_data",
            description="Run a SQL query on registered datasets using dataset('name') macro",
            guidance="Use SELECT only. Reference datasets with dataset('name'). "
            "Direct file access (read_parquet etc.) is blocked. "
            "Example: SELECT date, ret FROM dataset('equities') WHERE symbol='SPY' ORDER BY date",
        ),
        Tool(
            fit_model,
            name="fit_model",
            description="Fit a statistical model (OLS, ARIMA, GARCH, Markov) to data",
            guidance="Expert tool. Prefer analyze() for standard analyses — it returns richer output. "
            "Use fit_model for programmatic fits or when you need the raw RunRecord. "
            "Key params by family: GARCH: p, q; ARIMA: order; Markov: k_regimes.",
        ),
        Tool(
            forecast_model,
            name="forecast_model",
            description="Generate a forecast from a previously fitted model",
            guidance="Provide run_id from fit_model/analyze and number of steps to forecast.",
        ),
        Tool(
            run_diagnostics,
            name="run_diagnostics",
            description="Run stationarity tests (ADF + KPSS) on a data column",
            guidance="Call before fitting to check if data needs differencing.",
        ),
        Tool(
            get_run,
            name="get_run",
            description="Retrieve a past model run with full metrics and artifact paths",
        ),
        Tool(
            list_runs,
            name="list_runs",
            description="List past model runs, optionally filtered by dataset",
            guidance="Expert tool. Prefer discover_analyses() for enriched view "
            "with inline metrics and artifact catalogs.",
        ),
        Tool(
            compare_models,
            name="compare_models",
            description="Compare multiple model runs by AIC, BIC, and other metrics",
            guidance="Provide a list of run_ids. Returns a comparison with the best model.",
        ),
        Tool(
            list_artifacts,
            name="list_artifacts",
            description="List all artifacts (plots, data, summaries) for a model run",
            guidance="Expert tool. discover_analyses() includes artifacts inline. "
            "Use this for type-filtered artifact lookups.",
        ),
        Tool(
            get_plot,
            name="get_plot",
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
    from lazybridge.ext.doc_skills import build_skill

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


def stat_skill_tools(skill_dir_map: dict[str, Any]) -> list[Tool]:
    """Create Tool wrappers for the stat skill bundles."""
    from lazybridge.ext.doc_skills import skill_tool

    tools = []
    if "tool_guide" in skill_dir_map:
        tools.append(
            skill_tool(
                skill_dir_map["tool_guide"]["skill_dir"],
                name="stat_tool_guide",
                guidance="Query this when you need to know HOW to use a statistical tool — "
                "parameters, formats, workflows, error recovery.",
            )
        )
    if "quant_methodology" in skill_dir_map:
        tools.append(
            skill_tool(
                skill_dir_map["quant_methodology"]["skill_dir"],
                name="quant_methodology",
                guidance="Query this when you need to know WHAT to do — model selection, "
                "interpretation frameworks, risk analysis, reporting standards.",
            )
        )
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
    include_downloader: bool = True,
    expert_mode: bool = True,
    name: str = "stat_analyst",
    system: str | None = None,
    **agent_kwargs: Any,
) -> tuple[Any, Any]:
    """Create a pre-configured Agent with stat_runtime tools.

    Two operating modes:

    **expert_mode=True** (default):
        Main agent gets high-level tools (discover_data, discover_analyses,
        analyze, register_dataset) plus a ``delegate_to_expert`` tool that
        forwards tasks to an inner agent with all low-level tools.

    **expert_mode=False**:
        Main agent gets all tools in a flat list (no delegation).

    When **include_downloader=True** (default), adds data download tools:
        list_universe, search_tickers, download_tickers (Yahoo/FRED/ECB).

    Returns:
        (agent, runtime) tuple. The runtime is returned so the caller can
        register datasets programmatically or close it when done.

    Usage::

        agent, rt = stat_agent("anthropic")
        agent("Download SPY data and analyze its volatility")
        rt.close()
    """
    from lazybridge import Agent
    from lazybridge.engines.llm import LLMEngine
    from lazybridge.ext.stat_runtime.runner import StatRuntime

    rt = StatRuntime(db=db, artifacts_dir=artifacts_dir)
    model_str = model or (provider if isinstance(provider, str) else "anthropic")

    default_system = (
        "You are a quantitative analyst. Follow this workflow:\n"
        "1. search_tickers() or list_universe() — find tickers to analyze\n"
        "2. download_tickers() — download real market data\n"
        "3. discover_data() — verify registered datasets and column roles\n"
        "4. analyze() — run goal-oriented analysis (mode=recommend/volatility/forecast/regime)\n"
        "5. discover_analyses() — review completed analyses\n\n"
        "Always check model_adequate in results before trusting them. "
        "Report artifact paths so the user can view plots."
    )

    all_tools: list[Tool] = []

    # Add downloader tools if available
    if include_downloader:
        try:
            from lazybridge.ext.data_downloader.downloader import DataDownloader
            from lazybridge.ext.data_downloader.ticker_db import TickerDatabase
            from lazybridge.ext.data_downloader.tools import downloader_tools

            tdb = TickerDatabase()
            dl = DataDownloader(
                cache_dir=str(Path(artifacts_dir) / "ticker_cache"),
            )
            all_tools.extend(downloader_tools(rt, tdb, dl))
            _logger.info("Data downloader tools loaded (%d tickers)", len(tdb.list_all()))
        except Exception:
            _logger.warning("Data downloader not available", exc_info=True)

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

        _expert_kwargs = {k: v for k, v in agent_kwargs.items() if k not in ("system",)}
        expert = Agent(
            engine=LLMEngine(
                model_str,
                system="You are an expert statistical analyst. Use the available tools "
                "to fulfill the task precisely. Return results as structured data.",
            ),
            name="stat_expert",
            description="Expert statistical analyst with low-level tool access",
            tools=expert_tools,  # type: ignore[arg-type]
            **_expert_kwargs,
        )

        # Wrap expert as a tool for the main agent
        expert_tool = expert.as_tool(
            name="delegate_to_expert",
            description="Delegate a task to the expert statistical agent for fine-grained control. "
            "Use when the user needs specific model parameters, custom SQL queries, "
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

    _agent_kwargs = {k: v for k, v in agent_kwargs.items() if k not in ("system",)}
    agent = Agent(
        engine=LLMEngine(model_str, system=system or default_system),
        name=name,
        description="Statistical analyst with quantitative analysis capabilities",
        tools=all_tools,  # type: ignore[arg-type]
        **_agent_kwargs,
    )

    return agent, rt
