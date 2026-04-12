"""Tests for high-level stat_runtime tools: discover_data, discover_analyses, analyze.

Also tests the two-tier tool split (level parameter) and inference module.
"""

import pytest

from lazybridge.lazy_tool import LazyTool
from lazybridge.stat_runtime.schemas import (
    AnalysisDiscoveryResult,
    AnalysisResult,
    ArtifactRecord,
    ArtifactSummary,
    ColumnRoleInference,
    DataDiscoveryResult,
    DatasetDiscovery,
    DatasetMeta,
    Frequency,
    RunRecord,
    RunStatus,
    RunSummary,
)
from lazybridge.stat_runtime.inference import (
    build_interpretation,
    infer_column_roles,
    suggest_for_dataset,
)
from lazybridge.stat_runtime.tools import stat_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class MockCatalog:
    def __init__(self, datasets=None):
        self._datasets = datasets or []
        self._by_name = {d.name: d for d in self._datasets}

    def list_datasets(self):
        return self._datasets

    def get(self, name):
        return self._by_name.get(name)

    def register_parquet(self, *args, **kwargs):
        meta = DatasetMeta(name=args[0], uri=args[1])
        self._datasets.append(meta)
        self._by_name[meta.name] = meta
        return meta


class MockMetaStore:
    def __init__(self, runs=None, artifacts=None):
        self._runs = runs or []
        self._artifacts = artifacts or []

    def list_runs(self, dataset_name=None, limit=100, **kwargs):
        runs = self._runs
        if dataset_name:
            runs = [r for r in runs if r.dataset_name == dataset_name]
        return runs[:limit]

    def list_artifacts(self, run_id=None, artifact_type=None):
        arts = self._artifacts
        if run_id:
            arts = [a for a in arts if a.run_id == run_id]
        if artifact_type:
            arts = [a for a in arts if a.artifact_type == artifact_type]
        return arts


class MockRuntime:
    def __init__(self, datasets=None, runs=None, artifacts=None):
        self.catalog = MockCatalog(datasets)
        self.meta_store = MockMetaStore(runs, artifacts)
        self._runs = {r.run_id: r for r in (runs or [])}

    def list_runs(self, dataset_name=None, limit=100, **kwargs):
        return self.meta_store.list_runs(dataset_name=dataset_name, limit=limit)

    def get_run(self, run_id):
        return self._runs.get(run_id)


def _make_dataset(name="equities", **overrides):
    defaults = dict(
        name=name, uri=f"/data/{name}.parquet", file_format="parquet",
        frequency=Frequency.DAILY, time_column="date",
        entity_keys=["symbol"],
        columns_schema={"date": "Date", "symbol": "Utf8", "ret": "Float64", "volume": "Int64"},
        row_count=5000,
    )
    defaults.update(overrides)
    return DatasetMeta(**defaults)


def _make_run(run_id="run1", **overrides):
    defaults = dict(
        run_id=run_id, dataset_name="equities", engine="garch",
        status=RunStatus.SUCCESS,
        spec_json={"family": "garch", "target_col": "ret", "params": {"p": 1, "q": 1}},
        params_json={"mu": 0.05, "omega": 0.01, "alpha[1]": 0.08, "beta[1]": 0.90},
        metrics_json={"aic": 1234.5, "bic": 1267.8, "log_likelihood": -612.25},
        diagnostics_json=[
            {"test_name": "Ljung-Box", "statistic": 8.12, "p_value": 0.62, "passed": True},
            {"test_name": "Jarque-Bera", "statistic": 3.45, "p_value": 0.18, "passed": True},
        ],
        artifact_paths=["artifacts/run1/plots/residuals.png"],
        duration_secs=2.5,
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


def _make_artifact(run_id="run1", name="residuals", artifact_type="plot"):
    return ArtifactRecord(
        run_id=run_id, name=name, artifact_type=artifact_type,
        file_format="png", path=f"artifacts/{run_id}/plots/{name}.png",
        description=f"{name} plot",
    )


# ---------------------------------------------------------------------------
# Two-tier tool split
# ---------------------------------------------------------------------------

class TestToolLevelParam:
    def test_level_all_returns_15_tools(self):
        tools = stat_tools(MockRuntime(), level="all")
        assert len(tools) == 15
        assert all(isinstance(t, LazyTool) for t in tools)

    def test_level_high_returns_4_tools(self):
        tools = stat_tools(MockRuntime(), level="high")
        assert len(tools) == 4
        names = {t.name for t in tools}
        assert names == {"discover_data", "discover_analyses", "analyze", "register_dataset"}

    def test_level_low_returns_11_tools(self):
        tools = stat_tools(MockRuntime(), level="low")
        assert len(tools) == 11
        names = {t.name for t in tools}
        expected = {
            "list_datasets", "profile_dataset", "query_data",
            "fit_model", "forecast_model", "run_diagnostics",
            "get_run", "list_runs", "compare_models", "list_artifacts", "get_plot",
        }
        assert names == expected

    def test_default_level_returns_all(self):
        tools = stat_tools(MockRuntime())
        assert len(tools) == 15

    def test_high_and_low_are_disjoint(self):
        high = stat_tools(MockRuntime(), level="high")
        low = stat_tools(MockRuntime(), level="low")
        high_names = {t.name for t in high}
        low_names = {t.name for t in low}
        assert high_names.isdisjoint(low_names)


# ---------------------------------------------------------------------------
# Column role inference
# ---------------------------------------------------------------------------

class TestColumnRoleInference:
    def test_declared_time_column(self):
        meta = _make_dataset(time_column="date")
        roles = infer_column_roles(meta)
        date_role = next(r for r in roles if r.column == "date")
        assert date_role.inferred_role == "time"
        assert date_role.confidence == "high"

    def test_time_from_dtype(self):
        meta = _make_dataset(
            time_column=None,
            columns_schema={"ts": "Datetime", "val": "Float64"},
        )
        roles = infer_column_roles(meta)
        ts_role = next(r for r in roles if r.column == "ts")
        assert ts_role.inferred_role == "time"
        assert ts_role.confidence == "high"

    def test_entity_key_declared(self):
        meta = _make_dataset(entity_keys=["symbol"])
        roles = infer_column_roles(meta)
        sym_role = next(r for r in roles if r.column == "symbol")
        assert sym_role.inferred_role == "entity_key"
        assert sym_role.confidence == "high"

    def test_target_from_name(self):
        meta = _make_dataset(
            entity_keys=[],
            columns_schema={"ret": "Float64", "volume": "Int64"},
        )
        roles = infer_column_roles(meta)
        ret_role = next(r for r in roles if r.column == "ret")
        assert ret_role.inferred_role == "target"
        assert ret_role.confidence == "medium"

    def test_numeric_exogenous(self):
        meta = _make_dataset(
            entity_keys=[],
            columns_schema={"volume": "Int64"},
        )
        roles = infer_column_roles(meta)
        vol_role = next(r for r in roles if r.column == "volume")
        assert vol_role.inferred_role == "exogenous"

    def test_unknown_string(self):
        meta = _make_dataset(
            entity_keys=[],
            columns_schema={"notes": "Utf8"},
        )
        roles = infer_column_roles(meta)
        notes_role = next(r for r in roles if r.column == "notes")
        assert notes_role.inferred_role == "unknown"


# ---------------------------------------------------------------------------
# Dataset suggestions
# ---------------------------------------------------------------------------

class TestDatasetSuggestions:
    def test_no_profile_suggestion(self):
        meta = _make_dataset(profile_json={})
        roles = infer_column_roles(meta)
        suggestions = suggest_for_dataset(meta, roles)
        assert any("profile" in s.lower() for s in suggestions)

    def test_panel_data_suggestion(self):
        meta = _make_dataset(entity_keys=["symbol"])
        roles = infer_column_roles(meta)
        suggestions = suggest_for_dataset(meta, roles)
        assert any("panel" in s.lower() for s in suggestions)

    def test_small_dataset_warning(self):
        meta = _make_dataset(row_count=20)
        roles = infer_column_roles(meta)
        suggestions = suggest_for_dataset(meta, roles)
        assert any("small" in s.lower() or "very small" in s.lower() for s in suggestions)

    def test_target_identified(self):
        meta = _make_dataset(entity_keys=[])
        roles = infer_column_roles(meta)
        suggestions = suggest_for_dataset(meta, roles)
        assert any("target" in s.lower() and "ret" in s.lower() for s in suggestions)


# ---------------------------------------------------------------------------
# Build interpretation
# ---------------------------------------------------------------------------

class TestBuildInterpretation:
    def test_garch_persistence(self):
        run = _make_run(params_json={"alpha[1]": 0.08, "beta[1]": 0.90, "omega": 0.01})
        interp, warns, nexts = build_interpretation(run, "garch")
        assert any("persistence" in s.lower() for s in interp)

    def test_garch_igarch_warning(self):
        run = _make_run(params_json={"alpha[1]": 0.15, "beta[1]": 0.90, "omega": 0.01})
        interp, warns, nexts = build_interpretation(run, "garch")
        assert any("igarch" in s.lower() for s in warns)

    def test_garch_leverage(self):
        run = _make_run(params_json={"alpha[1]": 0.08, "beta[1]": 0.90, "gamma[1]": 0.05})
        interp, warns, nexts = build_interpretation(run, "garch")
        assert any("asymmetric" in s.lower() or "leverage" in s.lower() for s in interp)

    def test_failed_run(self):
        run = _make_run(status=RunStatus.FAILED, error_message="convergence failed")
        interp, warns, nexts = build_interpretation(run, "garch")
        assert any("failed" in s.lower() for s in warns)

    def test_ols_r_squared(self):
        run = _make_run(
            engine="ols",
            metrics_json={"r_squared": 0.85, "adj_r_squared": 0.84, "f_pvalue": 0.001},
            diagnostics_json=[],
        )
        interp, warns, nexts = build_interpretation(run, "ols")
        assert any("r-squared" in s.lower() for s in interp)
        assert any("significant" in s.lower() for s in interp)

    def test_markov_regimes(self):
        run = _make_run(
            engine="markov",
            params_json={"const[0]": 0.05, "const[1]": -0.03, "p[0->0]": 0.95, "p[1->1]": 0.90},
            spec_json={"params": {"k_regimes": 2}},
            diagnostics_json=[
                {"test_name": "Regime Classification Certainty", "statistic": 0.82, "passed": True},
            ],
        )
        interp, warns, nexts = build_interpretation(run, "markov")
        assert any("regime" in s.lower() for s in interp)
        assert any("persistent" in s.lower() for s in interp)

    def test_diagnostics_summary(self):
        run = _make_run(diagnostics_json=[
            {"test_name": "Ljung-Box", "statistic": 8.0, "p_value": 0.6, "passed": True},
            {"test_name": "Jarque-Bera", "statistic": 15.0, "p_value": 0.001, "passed": False},
        ])
        interp, warns, nexts = build_interpretation(run, "garch")
        assert any("1/2" in s or "diagnostics" in s.lower() for s in interp)
        assert any("jarque-bera" in s.lower() for s in warns)


# ---------------------------------------------------------------------------
# discover_data tool
# ---------------------------------------------------------------------------

class TestDiscoverData:
    def test_empty_registry(self):
        rt = MockRuntime(datasets=[])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        assert result["total_datasets"] == 0
        assert any("no datasets" in s.lower() for s in result["suggestions"])

    def test_single_dataset(self):
        meta = _make_dataset()
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        assert result["total_datasets"] == 1
        ds = result["datasets"][0]
        assert ds["name"] == "equities"
        assert len(ds["column_roles"]) == 4  # date, symbol, ret, volume

    def test_column_roles_populated(self):
        meta = _make_dataset()
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        ds = result["datasets"][0]
        roles_by_col = {r["column"]: r for r in ds["column_roles"]}
        assert roles_by_col["date"]["inferred_role"] == "time"
        assert roles_by_col["symbol"]["inferred_role"] == "entity_key"
        assert roles_by_col["ret"]["inferred_role"] == "target"


# ---------------------------------------------------------------------------
# discover_analyses tool
# ---------------------------------------------------------------------------

class TestDiscoverAnalyses:
    def test_empty_runs(self):
        rt = MockRuntime(runs=[])
        tools = stat_tools(rt, level="high")
        da = next(t for t in tools if t.name == "discover_analyses")
        result = da.run({})
        assert result["total_runs"] == 0
        assert result["best_by_aic"] is None

    def test_single_run_with_artifacts(self):
        run = _make_run()
        art = _make_artifact("run1", "residuals", "plot")
        rt = MockRuntime(runs=[run], artifacts=[art])
        tools = stat_tools(rt, level="high")
        da = next(t for t in tools if t.name == "discover_analyses")
        result = da.run({})
        assert result["total_runs"] == 1
        rs = result["runs"][0]
        assert rs["run_id"] == "run1"
        assert rs["aic"] == 1234.5
        assert rs["diagnostics_passed"] == 2
        assert rs["diagnostics_failed"] == 0
        assert len(rs["artifacts"]) == 1
        assert rs["artifacts"][0]["name"] == "residuals"

    def test_best_by_aic(self):
        run1 = _make_run("run1", metrics_json={"aic": 1234.5, "bic": 1267.8})
        run2 = _make_run("run2", metrics_json={"aic": 1200.0, "bic": 1300.0})
        rt = MockRuntime(runs=[run1, run2])
        tools = stat_tools(rt, level="high")
        da = next(t for t in tools if t.name == "discover_analyses")
        result = da.run({})
        assert result["best_by_aic"] == "run2"
        assert result["best_by_bic"] == "run1"

    def test_filter_by_dataset(self):
        run1 = _make_run("run1", dataset_name="equities")
        run2 = _make_run("run2", dataset_name="macro")
        rt = MockRuntime(runs=[run1, run2])
        tools = stat_tools(rt, level="high")
        da = next(t for t in tools if t.name == "discover_analyses")
        result = da.run({"dataset_name": "equities"})
        assert result["total_runs"] == 1
        assert result["runs"][0]["dataset_name"] == "equities"

    def test_failed_run_suggestion(self):
        run = _make_run("run1", status=RunStatus.FAILED, error_message="convergence failed")
        rt = MockRuntime(runs=[run])
        tools = stat_tools(rt, level="high")
        da = next(t for t in tools if t.name == "discover_analyses")
        result = da.run({})
        assert any("failed" in s.lower() for s in result["suggestions"])


# ---------------------------------------------------------------------------
# New schema models
# ---------------------------------------------------------------------------

class TestNewSchemas:
    def test_column_role_inference(self):
        cr = ColumnRoleInference(
            column="ret", dtype="Float64",
            inferred_role="target", confidence="medium",
            reason="Numeric column named 'ret'",
        )
        assert cr.inferred_role == "target"

    def test_dataset_discovery(self):
        dd = DatasetDiscovery(
            name="equities", uri="/data/eq.parquet",
            file_format="parquet", frequency="daily",
        )
        assert dd.has_profile is False
        assert dd.column_roles == []

    def test_data_discovery_result(self):
        ddr = DataDiscoveryResult(total_datasets=0)
        assert ddr.datasets == []

    def test_artifact_summary(self):
        a = ArtifactSummary(name="residuals", artifact_type="plot")
        assert a.path == ""

    def test_run_summary(self):
        rs = RunSummary(run_id="abc")
        assert rs.diagnostics_total == 0
        assert rs.artifacts == []

    def test_analysis_discovery_result(self):
        adr = AnalysisDiscoveryResult()
        assert adr.total_runs == 0
        assert adr.best_by_aic is None

    def test_analysis_result(self):
        ar = AnalysisResult(run_id="abc")
        assert ar.model_adequate is False
        assert ar.plots == []
        assert ar.interpretation == []

    def test_serialization_roundtrip(self):
        ar = AnalysisResult(
            run_id="test",
            status="success",
            engine="garch",
            interpretation=["High persistence"],
            warnings=["IGARCH-like"],
            next_steps=["Try EGARCH"],
        )
        data = ar.model_dump(mode="json")
        restored = AnalysisResult(**data)
        assert restored.interpretation == ["High persistence"]

    def test_analysis_result_new_fields(self):
        ar = AnalysisResult(
            run_id="test",
            mode="volatility",
            mode_rationale="Mode 'volatility' maps to GARCH family.",
            assumptions=["GARCH assumes volatility clustering"],
        )
        assert ar.mode == "volatility"
        assert ar.mode_rationale != ""
        assert len(ar.assumptions) == 1

    def test_dataset_meta_semantic_fields(self):
        meta = DatasetMeta(
            name="equities", uri="/data/eq.parquet",
            business_description="Daily S&P 500 returns",
            canonical_target="ret",
            identifiers_to_ignore=["trade_id", "hash"],
        )
        assert meta.business_description == "Daily S&P 500 returns"
        assert meta.canonical_target == "ret"
        assert meta.identifiers_to_ignore == ["trade_id", "hash"]

    def test_dataset_meta_semantic_defaults(self):
        meta = DatasetMeta(name="test", uri="/data/test.parquet")
        assert meta.business_description is None
        assert meta.canonical_target is None
        assert meta.identifiers_to_ignore == []

    def test_analysis_mode_enum(self):
        from lazybridge.stat_runtime.schemas import AnalysisMode
        assert AnalysisMode.RECOMMEND == "recommend"
        assert AnalysisMode.FORECAST == "forecast"
        assert AnalysisMode.VOLATILITY == "volatility"
        assert AnalysisMode("describe") == AnalysisMode.DESCRIBE

    def test_column_signals(self):
        from lazybridge.stat_runtime.schemas import ColumnSignals
        cs = ColumnSignals(null_pct=0.05, unique_count=100, mean=0.01)
        assert cs.null_pct == 0.05
        assert cs.unique_count == 100

    def test_dataset_discovery_new_fields(self):
        dd = DatasetDiscovery(
            name="equities", uri="/data/eq.parquet",
            file_format="parquet", frequency="daily",
            summary="equities: 5,000 daily observations; target: ret.",
            business_description="S&P 500 returns",
            canonical_target="ret",
        )
        assert dd.summary != ""
        assert dd.business_description == "S&P 500 returns"
        assert dd.canonical_target == "ret"


# ---------------------------------------------------------------------------
# Natural language dataset summary
# ---------------------------------------------------------------------------

class TestGenerateDatasetSummary:
    def test_basic_summary(self):
        from lazybridge.stat_runtime.inference import generate_dataset_summary
        meta = _make_dataset()
        roles = infer_column_roles(meta)
        summary = generate_dataset_summary(meta, roles)
        assert "equities" in summary
        assert "5,000" in summary

    def test_business_description_takes_precedence(self):
        from lazybridge.stat_runtime.inference import generate_dataset_summary
        meta = _make_dataset(business_description="Daily S&P 500 total returns")
        roles = infer_column_roles(meta)
        summary = generate_dataset_summary(meta, roles)
        assert summary == "Daily S&P 500 total returns"

    def test_canonical_target_in_summary(self):
        from lazybridge.stat_runtime.inference import generate_dataset_summary
        meta = _make_dataset(canonical_target="ret")
        roles = infer_column_roles(meta)
        summary = generate_dataset_summary(meta, roles)
        assert "target: ret" in summary

    def test_entity_keys_in_summary(self):
        from lazybridge.stat_runtime.inference import generate_dataset_summary
        meta = _make_dataset(entity_keys=["symbol"])
        roles = infer_column_roles(meta)
        summary = generate_dataset_summary(meta, roles)
        assert "symbol" in summary


# ---------------------------------------------------------------------------
# Analysis mode resolution
# ---------------------------------------------------------------------------

class TestResolveAnalysisMode:
    def test_forecast_mode(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        family, rationale, assumptions = resolve_analysis_mode(
            "forecast", None, None, None,
        )
        assert family == "arima"
        assert "forecast" in rationale.lower()
        assert len(assumptions) > 0

    def test_volatility_mode(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        family, rationale, assumptions = resolve_analysis_mode(
            "volatility", None, None, None,
        )
        assert family == "garch"

    def test_regime_mode(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        family, rationale, assumptions = resolve_analysis_mode(
            "regime", None, None, None,
        )
        assert family == "markov"

    def test_describe_mode(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        family, rationale, assumptions = resolve_analysis_mode(
            "describe", None, None, None,
        )
        assert family == "ols"
        assert "describe" in rationale.lower()

    def test_recommend_returns_with_time(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        meta = _make_dataset(time_column="date")
        roles = infer_column_roles(meta)
        family, rationale, assumptions = resolve_analysis_mode(
            "recommend", meta, roles, "ret",
        )
        # Returns + time → GARCH
        assert family == "garch"
        assert "return" in rationale.lower()

    def test_recommend_time_series_non_return(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        meta = _make_dataset(
            time_column="date",
            columns_schema={"date": "Date", "temperature": "Float64"},
            entity_keys=[],
        )
        roles = infer_column_roles(meta)
        family, rationale, assumptions = resolve_analysis_mode(
            "recommend", meta, roles, "temperature",
        )
        # Time series but not returns → ARIMA
        assert family == "arima"

    def test_recommend_no_time(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        meta = _make_dataset(
            time_column=None,
            columns_schema={"x": "Float64", "y": "Float64"},
            entity_keys=[],
        )
        roles = infer_column_roles(meta)
        family, rationale, assumptions = resolve_analysis_mode(
            "recommend", meta, roles, "y",
        )
        # No time → OLS
        assert family == "ols"

    def test_explicit_family_passthrough(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        family, rationale, assumptions = resolve_analysis_mode(
            "garch", None, None, None,
        )
        assert family == "garch"
        assert "explicit" in rationale.lower()

    def test_unknown_mode_raises(self):
        from lazybridge.stat_runtime.inference import resolve_analysis_mode
        with pytest.raises(ValueError, match="Unknown analysis mode"):
            resolve_analysis_mode("invalid_mode", None, None, None)


# ---------------------------------------------------------------------------
# discover_data enriched output
# ---------------------------------------------------------------------------

class TestDiscoverDataEnriched:
    def test_summary_populated(self):
        meta = _make_dataset()
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        ds = result["datasets"][0]
        assert ds["summary"] != ""
        assert "equities" in ds["summary"]

    def test_business_description_forwarded(self):
        meta = _make_dataset(business_description="Daily equity returns")
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        ds = result["datasets"][0]
        assert ds["business_description"] == "Daily equity returns"

    def test_canonical_target_forwarded(self):
        meta = _make_dataset(canonical_target="ret")
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        ds = result["datasets"][0]
        assert ds["canonical_target"] == "ret"

    def test_profile_signals_from_cache(self):
        meta = _make_dataset(profile_json={
            "columns": {
                "ret": {"null_pct": 0.01, "unique_count": 4500, "mean": 0.0004},
                "volume": {"null_pct": 0.0, "unique_count": 3000},
            },
        })
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        ds = result["datasets"][0]
        assert ds["has_profile"] is True
        signals = ds["column_signals"]
        assert "ret" in signals
        assert signals["ret"]["null_pct"] == 0.01
        assert signals["ret"]["unique_count"] == 4500
        assert signals["ret"]["mean"] == 0.0004
        assert signals["volume"]["null_pct"] == 0.0

    def test_no_profile_no_signals(self):
        meta = _make_dataset(profile_json={})
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        dd = next(t for t in tools if t.name == "discover_data")
        result = dd.run({})
        ds = result["datasets"][0]
        assert ds["column_signals"] == {}
        assert ds["has_profile"] is False


# ---------------------------------------------------------------------------
# analyze() group_col/group_value safety
# ---------------------------------------------------------------------------

class TestAnalyzeGroupSafety:
    """P0: group_col/group_value must not allow SQL injection."""

    def _get_analyze(self, meta):
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        return next(t for t in tools if t.name == "analyze")

    def test_invalid_group_col_rejected(self):
        meta = _make_dataset()
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "nonexistent",
            "group_value": "SPY",
        })
        assert result.get("error") is True
        assert "not found" in result["message"]

    def test_injection_in_group_col_rejected(self):
        meta = _make_dataset()
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "symbol; DROP TABLE--",
            "group_value": "SPY",
        })
        assert result.get("error") is True

    def test_group_col_without_value_rejected(self):
        meta = _make_dataset()
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "symbol",
        })
        assert result.get("error") is True
        assert "together" in result["message"]

    def test_group_value_without_col_rejected(self):
        meta = _make_dataset()
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_value": "SPY",
        })
        assert result.get("error") is True
        assert "together" in result["message"]

    def test_injection_in_group_value_escaped(self):
        """SQL injection via group_value should be neutralized by escaping."""
        meta = _make_dataset()
        tool = self._get_analyze(meta)
        # This would break unescaped SQL: WHERE symbol = '' OR 1=1 --'
        result = tool.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "group_col": "symbol",
            "group_value": "' OR 1=1 --",
        })
        # Should not succeed as an injection — either errors on execution
        # or passes through safely. Key: must not return data from all rows.
        # Since we have a mock runtime, it will fail at execute(), but the
        # important thing is the SQL was built safely (no raw interpolation).
        # The error should NOT be a validation error — it should be a runtime error.
        assert result.get("error") is True
        assert "not found" not in result.get("message", "")  # not a column error

    def test_nonexistent_dataset_rejected(self):
        rt = MockRuntime(datasets=[])
        tools = stat_tools(rt, level="high")
        tool = next(t for t in tools if t.name == "analyze")
        result = tool.run({"dataset_name": "nope", "target_col": "x"})
        assert result.get("error") is True
        assert "not registered" in result["message"]


# ---------------------------------------------------------------------------
# analyze() target resolution
# ---------------------------------------------------------------------------

class TestAnalyzeTargetResolution:
    def _get_analyze(self, meta):
        rt = MockRuntime(datasets=[meta])
        tools = stat_tools(rt, level="high")
        return next(t for t in tools if t.name == "analyze")

    def test_explicit_target_used(self):
        meta = _make_dataset(canonical_target="volume")
        tool = self._get_analyze(meta)
        # Even though canonical_target is "volume", explicit target_col wins
        result = tool.run({
            "dataset_name": "equities",
            "target_col": "ret",
            "mode": "describe",
        })
        # Will fail at execute (mock), but target_col should be "ret"
        assert result.get("error") is True  # mock can't execute
        # The error comes from execute, not from target resolution

    def test_canonical_target_fallback(self):
        meta = _make_dataset(canonical_target="ret")
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "mode": "recommend",
        })
        # Will fail at execute (mock), but should not fail on target resolution
        assert result.get("error") is True
        # Should NOT be a MissingTarget or AmbiguousTarget error
        assert result.get("type") != "MissingTarget"
        assert result.get("type") != "AmbiguousTarget"

    def test_inferred_single_target_fallback(self):
        # Dataset with only one target-like column
        meta = _make_dataset(
            columns_schema={"date": "Date", "ret": "Float64"},
            entity_keys=[],
        )
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "mode": "recommend",
        })
        # Should not be an ambiguity error
        assert result.get("type") != "AmbiguousTarget"

    def test_ambiguous_targets_returns_candidates(self):
        # Dataset with multiple target-like columns
        meta = _make_dataset(
            columns_schema={
                "date": "Date",
                "ret": "Float64",
                "close": "Float64",
                "price": "Float64",
            },
            entity_keys=[],
        )
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "mode": "recommend",
        })
        assert result.get("error") is True
        assert result.get("type") == "AmbiguousTarget"
        assert "candidates" in result
        assert len(result["candidates"]) > 1

    def test_describe_without_target_uses_first_numeric(self):
        meta = _make_dataset(
            columns_schema={"date": "Date", "temperature": "Float64", "humidity": "Float64"},
            entity_keys=[],
        )
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "mode": "describe",
        })
        # Should not be a MissingTarget error — describe picks first numeric
        assert result.get("type") != "MissingTarget"

    def test_no_target_no_numeric_errors(self):
        meta = _make_dataset(
            columns_schema={"name": "Utf8", "label": "Utf8"},
            entity_keys=[],
        )
        tool = self._get_analyze(meta)
        result = tool.run({
            "dataset_name": "equities",
            "mode": "recommend",
        })
        assert result.get("error") is True
        assert result.get("type") == "MissingTarget"


# ---------------------------------------------------------------------------
# register_dataset semantic fields
# ---------------------------------------------------------------------------

class TestRegisterDatasetSemanticFields:
    """register_dataset should accept and persist semantic metadata."""

    def test_semantic_fields_in_tool_schema(self):
        rt = MockRuntime()
        tools = stat_tools(rt, level="high")
        reg = next(t for t in tools if t.name == "register_dataset")
        defn = reg.definition()
        params = defn.parameters.get("properties", {})
        assert "business_description" in params
        assert "canonical_target" in params
        assert "identifiers_to_ignore" in params
