"""RunManager — orchestrates fit, diagnostics, plots, and persistence.

This is the main entry point for executing statistical workflows.

Usage::

    from lazybridge.stat_runtime.runner import StatRuntime

    rt = StatRuntime()
    rt.catalog.register_parquet("equities", "data/returns.parquet", time_column="date")

    run = rt.execute(ModelSpec(
        family="garch", target_col="ret", dataset_name="equities",
        params={"p": 1, "q": 1}, forecast_steps=20,
    ))
    print(run.metrics_json)
    print(run.artifact_paths)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from lazybridge.stat_runtime.artifact_store import ArtifactStore
from lazybridge.stat_runtime.catalog import DatasetCatalog
from lazybridge.stat_runtime.engines import get_engine
from lazybridge.stat_runtime.persistence import MetaStore
from lazybridge.stat_runtime.query import QueryEngine
from lazybridge.stat_runtime.schemas import (
    DiagnosticResult,
    FitResult,
    ForecastResult,
    ModelFamily,
    ModelSpec,
    RunRecord,
    RunStatus,
)

_logger = logging.getLogger(__name__)


class StatRuntime:
    """Statistical runtime — the main entry point.

    Orchestrates data access, model fitting, diagnostics, visualization,
    and result persistence.

    Usage::

        rt = StatRuntime()                                      # in-memory
        rt = StatRuntime(db="my.duckdb", artifacts_dir="out")  # persistent

        with StatRuntime(db="my.duckdb") as rt:
            rt.catalog.register_parquet("data", "file.parquet")
            run = rt.execute(spec)
    """

    def __init__(
        self,
        *,
        db: str | None = None,
        artifacts_dir: str = "artifacts",
    ) -> None:
        self.meta_store = MetaStore(db=db)
        self.artifacts = ArtifactStore(root=artifacts_dir, meta_store=self.meta_store)
        self.catalog = DatasetCatalog(self.meta_store)
        self.query_engine = QueryEngine(self.catalog)

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def execute(self, spec: ModelSpec) -> RunRecord:
        """Fit a model, run diagnostics, generate plots, and persist results.

        Never raises — catches all exceptions and returns a RunRecord with
        status=FAILED and error_message set.
        """
        run = RunRecord(
            dataset_name=spec.dataset_name or "",
            spec_json=spec.model_dump(),
            engine=str(spec.family),
        )
        start = time.monotonic()

        try:
            run.status = RunStatus.RUNNING
            self.meta_store.save_run(run)

            # 1. Load data
            y, X = self._load_data(spec)

            # 2. Fit model
            engine = get_engine(spec.family)
            fit_result = engine.fit(y, X, spec=spec)

            # 3. Run diagnostics
            diag_results = engine.diagnostics(fit_result, y, X)

            # 4. Generate forecast if requested
            forecast_result = None
            if spec.forecast_steps and spec.forecast_steps > 0:
                forecast_result = engine.forecast(fit_result, spec.forecast_steps)

            # 5. Generate plots
            artifact_paths = self._generate_plots(
                run.run_id, spec, fit_result, y, forecast_result,
            )

            # 6. Save data artifacts
            artifact_paths.extend(self._save_data_artifacts(run.run_id, fit_result, forecast_result))

            # 7. Save summary artifacts
            self.artifacts.write_json(
                run.run_id, "spec", spec.model_dump(), artifact_type="summary",
                description="Model specification",
            )
            self.artifacts.write_text(
                run.run_id, "fit_summary", fit_result.summary_text, artifact_type="summary",
                file_format="txt", description="Model fit summary",
            )
            self.artifacts.write_json(
                run.run_id, "diagnostics",
                [d.model_dump() for d in diag_results],
                artifact_type="summary", description="Diagnostic test results",
            )
            artifact_paths.extend([
                str(self.artifacts.path_for(run.run_id, "spec", "summary", ".json")),
                str(self.artifacts.path_for(run.run_id, "fit_summary", "summary", ".txt")),
                str(self.artifacts.path_for(run.run_id, "diagnostics", "summary", ".json")),
            ])

            # 8. Update run record
            run.status = RunStatus.SUCCESS
            run.fit_summary = fit_result.summary_text[:2000]
            run.params_json = fit_result.params
            run.metrics_json = fit_result.metrics
            run.diagnostics_json = [d.model_dump() for d in diag_results]
            run.artifact_paths = artifact_paths

        except Exception as exc:
            _logger.exception("Run %s failed: %s", run.run_id, exc)
            run.status = RunStatus.FAILED
            run.error_message = f"{type(exc).__name__}: {exc}"

        run.duration_secs = round(time.monotonic() - start, 3)
        self.meta_store.save_run(run)
        return run

    # ------------------------------------------------------------------
    # Forecast from existing run
    # ------------------------------------------------------------------

    def forecast(self, run_id: str, steps: int, *, ci_level: float = 0.95) -> ForecastResult:
        """Generate a forecast from a previously fitted model.

        NOTE: This requires the fitted model object to be in memory
        (i.e., the run must have been executed in this session).
        """
        run = self.meta_store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run '{run_id}' not found")
        if run.status != RunStatus.SUCCESS:
            raise ValueError(f"Run '{run_id}' did not succeed (status={run.status})")

        engine = get_engine(run.engine)
        # Reconstruct FitResult from stored data
        fit_result = FitResult(
            family=ModelFamily(run.engine),
            summary_text=run.fit_summary,
            params=run.params_json,
            metrics=run.metrics_json,
        )
        return engine.forecast(fit_result, steps, ci_level=ci_level)

    # ------------------------------------------------------------------
    # Query passthrough
    # ------------------------------------------------------------------

    def query(self, sql: str, *, max_rows: int = 10_000):
        """Execute a SQL query against registered datasets."""
        return self.query_engine.execute(sql, max_rows=max_rows)

    # ------------------------------------------------------------------
    # Run retrieval
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.meta_store.get_run(run_id)

    def list_runs(self, **kwargs) -> list[RunRecord]:
        return self.meta_store.list_runs(**kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_data(self, spec: ModelSpec) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract target and exog arrays from the dataset or query."""
        if spec.query_sql:
            result = self.query_engine.execute(spec.query_sql)
            data = result.data_json
        elif spec.dataset_name:
            pl = _import_polars()
            df = self.catalog.load_df(spec.dataset_name)
            data = df.to_dicts()
        else:
            raise ValueError("ModelSpec must have either dataset_name or query_sql")

        if not data:
            raise ValueError("Query returned no data")
        if spec.target_col not in data[0]:
            raise ValueError(
                f"Target column '{spec.target_col}' not found. "
                f"Available: {list(data[0].keys())}"
            )

        y = np.array([row[spec.target_col] for row in data], dtype=float)
        X = None
        if spec.exog_cols:
            X = np.column_stack([
                np.array([row[col] for row in data], dtype=float)
                for col in spec.exog_cols
            ])

        # Remove NaN rows
        mask = ~np.isnan(y)
        if X is not None:
            mask &= ~np.any(np.isnan(X), axis=1)
        y = y[mask]
        if X is not None:
            X = X[mask]

        return y, X

    def _generate_plots(
        self,
        run_id: str,
        spec: ModelSpec,
        fit_result: FitResult,
        y: np.ndarray,
        forecast_result: ForecastResult | None,
    ) -> list[str]:
        """Generate appropriate plots based on model family."""
        from lazybridge.stat_runtime import plots

        paths: list[str] = []
        try:
            # Residual plots (all models)
            if fit_result.residuals_json:
                paths.append(plots.plot_residuals(
                    fit_result.residuals_json, run_id, self.artifacts,
                ))
                paths.append(plots.plot_acf_pacf(
                    fit_result.residuals_json, run_id, self.artifacts,
                    title="Residual ACF & PACF",
                ))

            # Family-specific plots
            if spec.family == ModelFamily.GARCH:
                cond_vol = fit_result.extra.get("conditional_volatility")
                if cond_vol:
                    paths.append(plots.plot_volatility(
                        cond_vol, run_id, self.artifacts, returns=y.tolist(),
                    ))

            elif spec.family == ModelFamily.MARKOV:
                smoothed = fit_result.extra.get("smoothed_probabilities")
                if smoothed:
                    paths.append(plots.plot_regimes(
                        smoothed, run_id, self.artifacts, series=y.tolist(),
                    ))

            # Forecast plot
            if forecast_result is not None and forecast_result.point_forecast:
                paths.append(plots.plot_forecast(
                    y, forecast_result, run_id, self.artifacts,
                ))

        except Exception as exc:
            _logger.warning("Plot generation failed (non-fatal): %s", exc)

        return paths

    def _save_data_artifacts(
        self,
        run_id: str,
        fit_result: FitResult,
        forecast_result: ForecastResult | None,
    ) -> list[str]:
        """Save data artifacts (residuals, forecast data)."""
        paths: list[str] = []
        try:
            if fit_result.residuals_json:
                paths.append(self.artifacts.write_json(
                    run_id, "residuals", fit_result.residuals_json,
                    artifact_type="data", description="Model residuals",
                ))
            if forecast_result:
                paths.append(self.artifacts.write_json(
                    run_id, "forecast", forecast_result.model_dump(),
                    artifact_type="forecast", description="Forecast results",
                ))
        except Exception as exc:
            _logger.warning("Data artifact save failed (non-fatal): %s", exc)
        return paths

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.meta_store.close()

    def __enter__(self) -> StatRuntime:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"StatRuntime(meta_store={self.meta_store!r}, artifacts={self.artifacts.root})"


def _import_polars():
    from lazybridge.stat_runtime._deps import require_polars
    return require_polars()
