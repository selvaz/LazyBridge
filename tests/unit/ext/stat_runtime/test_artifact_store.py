"""Tests for ArtifactStore — filesystem only, no heavy deps."""

import pytest

from lazybridge.ext.stat_runtime.artifact_store import ArtifactStore
from lazybridge.ext.stat_runtime.persistence import MetaStore


@pytest.fixture
def store(tmp_path):
    meta = MetaStore()
    return ArtifactStore(root=str(tmp_path / "artifacts"), meta_store=meta), meta


class TestWriteAndRead:
    def test_write_json(self, store):
        art_store, meta = store
        path = art_store.write_json("run1", "spec", {"family": "garch"})
        assert path.endswith(".json")
        data = art_store.read_json("run1", "spec")
        assert data["family"] == "garch"

    def test_write_text(self, store):
        art_store, meta = store
        path = art_store.write_text("run1", "summary", "Model fit OK")
        assert path.endswith(".txt")
        content = art_store.read_bytes("run1", "summary", artifact_type="summary")
        assert b"Model fit OK" in content

    def test_write_bytes(self, store):
        art_store, meta = store
        path = art_store.write_bytes("run1", "raw", b"binary_data", file_format="bin")
        # Read using exact filename since .bin isn't in the auto-search list
        from pathlib import Path

        assert Path(path).read_bytes() == b"binary_data"

    def test_list_files(self, store):
        art_store, meta = store
        art_store.write_json("run1", "spec", {})
        art_store.write_text("run1", "notes", "test")
        files = art_store.list_files("run1")
        assert len(files) >= 2

    def test_list_files_empty(self, store):
        art_store, meta = store
        assert art_store.list_files("nonexistent") == []

    def test_exists(self, store):
        art_store, meta = store
        art_store.write_json("run1", "spec", {})
        assert art_store.exists("run1", "spec", "summary")
        assert not art_store.exists("run1", "missing", "summary")


class TestMetaRegistration:
    def test_artifacts_registered_in_meta(self, store):
        art_store, meta = store
        art_store.write_json("run1", "spec", {"test": True})
        arts = meta.list_artifacts(run_id="run1")
        assert len(arts) == 1
        assert arts[0].name == "spec"
        assert arts[0].artifact_type == "summary"

    def test_no_meta_store(self, tmp_path):
        art_store = ArtifactStore(root=str(tmp_path / "art"), meta_store=None)
        # Should work without meta store
        path = art_store.write_json("run1", "spec", {})
        assert path.endswith(".json")


class TestDirectoryLayout:
    def test_plots_in_plots_dir(self, store):
        art_store, meta = store
        path = art_store.path_for("run1", "test", "plot", ".png")
        assert "/plots/" in str(path)

    def test_summaries_in_summaries_dir(self, store):
        art_store, meta = store
        path = art_store.path_for("run1", "test", "summary", ".json")
        assert "/summaries/" in str(path)

    def test_data_in_data_dir(self, store):
        art_store, meta = store
        path = art_store.path_for("run1", "test", "data", ".csv")
        assert "/data/" in str(path)
