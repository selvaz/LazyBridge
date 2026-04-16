"""Tests for the InMemory persistence backend — no DuckDB required."""

import pytest

from lazybridge.ext.stat_runtime.persistence import MetaStore
from lazybridge.ext.stat_runtime.schemas import (
    ArtifactRecord,
    DatasetMeta,
    RunRecord,
    RunStatus,
)


@pytest.fixture
def store():
    return MetaStore()  # in-memory


class TestDatasetCRUD:
    def test_save_and_get(self, store):
        meta = DatasetMeta(name="test", uri="/data/test.parquet")
        store.save_dataset(meta)
        result = store.get_dataset("test")
        assert result is not None
        assert result.name == "test"
        assert result.uri == "/data/test.parquet"

    def test_get_missing(self, store):
        assert store.get_dataset("nonexistent") is None

    def test_list_datasets(self, store):
        store.save_dataset(DatasetMeta(name="a", uri="/a.parquet"))
        store.save_dataset(DatasetMeta(name="b", uri="/b.parquet"))
        datasets = store.list_datasets()
        assert len(datasets) == 2
        assert {d.name for d in datasets} == {"a", "b"}

    def test_delete_dataset(self, store):
        store.save_dataset(DatasetMeta(name="test", uri="/test.parquet"))
        store.delete_dataset("test")
        assert store.get_dataset("test") is None

    def test_overwrite_dataset(self, store):
        store.save_dataset(DatasetMeta(name="test", uri="/old.parquet"))
        store.save_dataset(DatasetMeta(name="test", uri="/new.parquet"))
        result = store.get_dataset("test")
        assert result.uri == "/new.parquet"


class TestRunCRUD:
    def test_save_and_get(self, store):
        run = RunRecord(dataset_name="test", engine="ols", status=RunStatus.SUCCESS)
        store.save_run(run)
        result = store.get_run(run.run_id)
        assert result is not None
        assert result.engine == "ols"

    def test_get_missing(self, store):
        assert store.get_run("nonexistent") is None

    def test_list_runs(self, store):
        store.save_run(RunRecord(dataset_name="a", engine="ols"))
        store.save_run(RunRecord(dataset_name="b", engine="garch"))
        store.save_run(RunRecord(dataset_name="a", engine="arima"))
        all_runs = store.list_runs()
        assert len(all_runs) == 3

    def test_list_runs_by_dataset(self, store):
        store.save_run(RunRecord(dataset_name="a", engine="ols"))
        store.save_run(RunRecord(dataset_name="b", engine="garch"))
        runs = store.list_runs(dataset_name="a")
        assert len(runs) == 1
        assert runs[0].dataset_name == "a"

    def test_list_runs_by_status(self, store):
        store.save_run(RunRecord(status=RunStatus.SUCCESS))
        store.save_run(RunRecord(status=RunStatus.FAILED))
        store.save_run(RunRecord(status=RunStatus.SUCCESS))
        runs = store.list_runs(status="success")
        assert len(runs) == 2

    def test_list_runs_limit(self, store):
        for _ in range(10):
            store.save_run(RunRecord())
        runs = store.list_runs(limit=3)
        assert len(runs) == 3


class TestArtifactCRUD:
    def test_save_and_list(self, store):
        art = ArtifactRecord(
            run_id="run1",
            name="volatility",
            artifact_type="plot",
            file_format="png",
            path="/artifacts/run1/plots/volatility.png",
        )
        store.save_artifact(art)
        arts = store.list_artifacts(run_id="run1")
        assert len(arts) == 1
        assert arts[0].name == "volatility"

    def test_filter_by_type(self, store):
        store.save_artifact(
            ArtifactRecord(
                run_id="run1",
                name="vol",
                artifact_type="plot",
                file_format="png",
                path="/a",
            )
        )
        store.save_artifact(
            ArtifactRecord(
                run_id="run1",
                name="spec",
                artifact_type="summary",
                file_format="json",
                path="/b",
            )
        )
        plots = store.list_artifacts(run_id="run1", artifact_type="plot")
        assert len(plots) == 1
        assert plots[0].name == "vol"


class TestContextManager:
    def test_with_statement(self):
        with MetaStore() as store:
            store.save_dataset(DatasetMeta(name="test", uri="/test.parquet"))
            assert store.get_dataset("test") is not None
