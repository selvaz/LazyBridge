"""Tests for stat_runtime schemas — pure Pydantic, no heavy deps needed."""

from lazybridge.external_tools.stat_runtime.schemas import (
    ArtifactRecord,
    ColumnProfile,
    DatasetMeta,
    DiagnosticResult,
    FitResult,
    ForecastResult,
    Frequency,
    ModelFamily,
    ModelSpec,
    ProfileResult,
    QueryResult,
    RunRecord,
    RunStatus,
)


class TestEnums:
    def test_model_family_values(self):
        assert ModelFamily.OLS == "ols"
        assert ModelFamily.ARIMA == "arima"
        assert ModelFamily.GARCH == "garch"
        assert ModelFamily.MARKOV == "markov"

    def test_run_status_values(self):
        assert RunStatus.PENDING == "pending"
        assert RunStatus.SUCCESS == "success"
        assert RunStatus.FAILED == "failed"

    def test_frequency_values(self):
        assert Frequency.DAILY == "daily"
        assert Frequency.INTRADAY == "intraday"
        assert Frequency.IRREGULAR == "irregular"

    def test_model_family_from_string(self):
        assert ModelFamily("garch") == ModelFamily.GARCH


class TestDatasetMeta:
    def test_defaults(self):
        meta = DatasetMeta(name="test", uri="/data/test.parquet")
        assert meta.name == "test"
        assert meta.file_format == "parquet"
        assert meta.frequency == Frequency.DAILY
        assert meta.columns_schema == {}
        assert meta.entity_keys == []
        assert meta.profile_json == {}
        assert meta.row_count is None
        assert len(meta.dataset_id) == 12

    def test_serialization_roundtrip(self):
        meta = DatasetMeta(
            name="equities",
            uri="/data/eq.parquet",
            time_column="date",
            entity_keys=["symbol"],
            columns_schema={"date": "Date", "ret": "Float64"},
        )
        data = meta.model_dump(mode="json")
        restored = DatasetMeta(**data)
        assert restored.name == "equities"
        assert restored.time_column == "date"
        assert restored.entity_keys == ["symbol"]


class TestModelSpec:
    def test_basic(self):
        spec = ModelSpec(
            family=ModelFamily.GARCH,
            target_col="ret",
            dataset_name="equities",
            params={"p": 1, "q": 1},
        )
        assert spec.family == ModelFamily.GARCH
        assert spec.params["p"] == 1

    def test_with_forecast(self):
        spec = ModelSpec(
            family="arima",
            target_col="price",
            forecast_steps=20,
        )
        assert spec.forecast_steps == 20

    def test_serialization(self):
        spec = ModelSpec(family="ols", target_col="y", exog_cols=["x1", "x2"])
        data = spec.model_dump(mode="json")
        assert data["family"] == "ols"
        assert data["exog_cols"] == ["x1", "x2"]


class TestFitResult:
    def test_defaults(self):
        fr = FitResult(family=ModelFamily.OLS)
        assert fr.params == {}
        assert fr.metrics == {}
        assert fr.residuals_json == []

    def test_with_data(self):
        fr = FitResult(
            family=ModelFamily.GARCH,
            params={"omega": 0.01, "alpha": 0.1, "beta": 0.85},
            metrics={"aic": 1234.5, "bic": 1256.7},
            residuals_json=[0.1, -0.2, 0.3],
            extra={"conditional_volatility": [0.5, 0.6, 0.7]},
        )
        assert fr.params["beta"] == 0.85
        assert len(fr.extra["conditional_volatility"]) == 3


class TestForecastResult:
    def test_basic(self):
        fc = ForecastResult(
            family=ModelFamily.ARIMA,
            steps=5,
            point_forecast=[1.0, 1.1, 1.2, 1.3, 1.4],
            lower_ci=[0.8, 0.85, 0.9, 0.95, 1.0],
            upper_ci=[1.2, 1.35, 1.5, 1.65, 1.8],
        )
        assert fc.steps == 5
        assert fc.ci_level == 0.95
        assert len(fc.point_forecast) == 5


class TestDiagnosticResult:
    def test_basic(self):
        diag = DiagnosticResult(
            test_name="ADF",
            statistic=-3.5,
            p_value=0.008,
            passed=True,
            interpretation="Series is stationary at 5% level.",
        )
        assert diag.passed is True
        assert diag.p_value < 0.05


class TestRunRecord:
    def test_defaults(self):
        run = RunRecord()
        assert run.status == RunStatus.PENDING
        assert run.dataset_name == ""
        assert len(run.run_id) == 16

    def test_full(self):
        run = RunRecord(
            dataset_name="equities",
            engine="garch",
            status=RunStatus.SUCCESS,
            params_json={"omega": 0.01},
            metrics_json={"aic": 1234.5},
            artifact_paths=["/artifacts/abc/plots/volatility.png"],
        )
        assert run.status == RunStatus.SUCCESS
        assert len(run.artifact_paths) == 1


class TestArtifactRecord:
    def test_basic(self):
        art = ArtifactRecord(
            run_id="abc123",
            name="volatility",
            artifact_type="plot",
            file_format="png",
            path="/artifacts/abc123/plots/volatility.png",
        )
        assert art.artifact_type == "plot"
        assert art.metadata == {}


class TestQueryResult:
    def test_basic(self):
        qr = QueryResult(
            query_hash="abc123",
            original_sql="SELECT * FROM t",
            normalized_sql="SELECT * FROM t",
            columns=["a", "b"],
            row_count=100,
        )
        assert qr.truncated is False
        assert qr.data_json == []


class TestProfileResult:
    def test_basic(self):
        pr = ProfileResult(
            dataset_name="test",
            row_count=1000,
            columns={
                "ret": ColumnProfile(dtype="Float64", null_count=5, null_pct=0.5),
            },
        )
        assert pr.columns["ret"].null_pct == 0.5
