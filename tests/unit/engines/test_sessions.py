"""SessionRegistry: the durable alias store shared by both coding engines."""

from __future__ import annotations

import json
import os
import threading
import warnings

import pytest

import lazybridge
from lazybridge.engines import sessions as sessions_mod
from lazybridge.engines.sessions import (
    SESSIONS_FILE_ENV,
    SessionRegistry,
    default_session_registry,
    normalize_scope,
    set_default_session_registry,
)


@pytest.fixture
def registry(tmp_path):
    return SessionRegistry(tmp_path / "sessions.json")


@pytest.fixture
def scope(tmp_path):
    return str(tmp_path / "repo")


class TestBasics:
    def test_missing_file_is_an_empty_registry(self, registry, scope):
        assert registry.resolve("codex", scope, "review") is None
        assert registry.entries() == []
        assert not registry.path.exists()

    def test_bind_then_resolve_and_entries(self, registry, scope):
        registry.bind("codex", scope, "review", "thr-1", model="gpt-6-luna", effort="low")

        assert registry.resolve("codex", scope, "review") == "thr-1"
        (row,) = registry.entries()
        assert row["kind"] == "codex"
        assert row["name"] == "review"
        assert row["native_id"] == "thr-1"
        assert row["model"] == "gpt-6-luna"
        assert row["effort"] == "low"
        assert row["created_at"] and row["updated_at"]

    def test_rebind_repoints_keeps_created_at_and_overwrites_model(self, registry, scope):
        registry.bind("claude", scope, "a", "s1", model="haiku", effort="low")
        first = registry.entries()[0]
        registry.bind("claude", scope, "a", "s2")

        assert registry.resolve("claude", scope, "a") == "s2"
        (row,) = registry.entries()
        assert row["created_at"] == first["created_at"]
        assert row["model"] is None and row["effort"] is None

    def test_kinds_and_names_are_independent(self, registry, scope):
        registry.bind("codex", scope, "x", "c1")
        registry.bind("claude", scope, "x", "k1")
        registry.bind("codex", scope, "y", "c2")

        assert registry.resolve("codex", scope, "x") == "c1"
        assert registry.resolve("claude", scope, "x") == "k1"
        assert [r["name"] for r in registry.entries("codex")] == ["x", "y"]
        assert len(registry.entries("claude")) == 1

    def test_the_file_is_plain_json_an_operator_can_edit(self, registry, scope):
        registry.bind("codex", scope, "review", "thr-1")
        data = json.loads(registry.path.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert data["sessions"]["codex"][normalize_scope(scope)]["review"]["native_id"] == "thr-1"

        # A hand edit is honoured on the next read.
        data["sessions"]["codex"][normalize_scope(scope)]["review"]["native_id"] = "thr-hand"
        registry.path.write_text(json.dumps(data), encoding="utf-8")
        assert registry.resolve("codex", scope, "review") == "thr-hand"

    def test_bind_rejects_an_empty_native_id(self, registry, scope):
        with pytest.raises(ValueError, match="native_id"):
            registry.bind("codex", scope, "a", "  ")

    def test_unknown_kind_is_rejected(self, registry, scope):
        with pytest.raises(ValueError, match="kind"):
            registry.bind("gemini", scope, "a", "x")


class TestScope:
    def test_scope_is_a_resolved_absolute_posix_path(self, tmp_path):
        normalized = normalize_scope(tmp_path / "a" / ".." / "b")
        assert "\\" not in normalized
        assert ".." not in normalized
        assert normalized.endswith("/b")

    @pytest.mark.skipif(os.name != "nt", reason="case-insensitive paths are a Windows rule")
    def test_scope_is_case_insensitive_on_windows(self, tmp_path):
        assert normalize_scope(str(tmp_path)) == normalize_scope(str(tmp_path).upper())

    def test_same_alias_in_two_scopes_are_two_sessions(self, registry, tmp_path):
        registry.bind("codex", tmp_path / "one", "review", "thr-1")
        registry.bind("codex", tmp_path / "two", "review", "thr-2")

        assert registry.resolve("codex", tmp_path / "one", "review") == "thr-1"
        assert registry.resolve("codex", tmp_path / "two", "review") == "thr-2"
        assert [r["native_id"] for r in registry.entries(scope=tmp_path / "two")] == ["thr-2"]

    def test_empty_scope_is_rejected(self, registry):
        with pytest.raises(ValueError, match="scope"):
            registry.resolve("codex", "", "a")


class TestLabels:
    @pytest.mark.parametrize("good", ["a", "review", "Code-Review_2.x", "a" * 64])
    def test_valid_labels(self, registry, scope, good):
        registry.bind("codex", scope, good, "id")
        assert registry.resolve("codex", scope, good) == "id"

    @pytest.mark.parametrize("bad", ["", "1abc", "-x", "has space", "a/b", "a" * 65, "é", ".hidden"])
    def test_invalid_labels_are_refused_with_a_clear_message(self, registry, scope, bad):
        with pytest.raises(ValueError, match="invalid session alias"):
            registry.bind("codex", scope, bad, "id")
        with pytest.raises(ValueError, match="invalid session alias"):
            registry.resolve("codex", scope, bad)

    def test_rename_validates_the_new_label(self, registry, scope):
        registry.bind("codex", scope, "a", "id")
        with pytest.raises(ValueError, match="invalid new session alias"):
            registry.rename("codex", scope, "a", "bad name")


class TestRenameForget:
    def test_rename_moves_the_entry(self, registry, scope):
        registry.bind("codex", scope, "old", "thr-1", model="m")
        registry.rename("codex", scope, "old", "new")

        assert registry.resolve("codex", scope, "old") is None
        assert registry.resolve("codex", scope, "new") == "thr-1"
        assert registry.entries()[0]["model"] == "m"

    def test_rename_refuses_to_overwrite_an_existing_target(self, registry, scope):
        registry.bind("codex", scope, "a", "thr-a")
        registry.bind("codex", scope, "b", "thr-b")

        with pytest.raises(ValueError, match="already exists"):
            registry.rename("codex", scope, "a", "b")
        assert registry.resolve("codex", scope, "a") == "thr-a"
        assert registry.resolve("codex", scope, "b") == "thr-b"

    def test_rename_of_a_missing_alias_is_a_key_error(self, registry, scope):
        with pytest.raises(KeyError):
            registry.rename("codex", scope, "ghost", "x")

    def test_rename_to_itself_is_a_no_op(self, registry, scope):
        registry.bind("codex", scope, "a", "thr-a")
        registry.rename("codex", scope, "a", "a")
        assert registry.resolve("codex", scope, "a") == "thr-a"

    def test_forget_reports_whether_it_existed(self, registry, scope):
        registry.bind("codex", scope, "a", "thr-a")

        assert registry.forget("codex", scope, "a") is True
        assert registry.forget("codex", scope, "a") is False
        assert registry.resolve("codex", scope, "a") is None
        assert registry.entries() == []

    def test_forget_on_a_missing_file_is_a_no_op(self, registry, scope):
        assert registry.forget("codex", scope, "a") is False
        assert not registry.path.exists()


class TestCorruptFile:
    def test_a_corrupt_file_is_quarantined_not_overwritten(self, registry, scope):
        registry.path.write_text("{ this is not json", encoding="utf-8")

        with pytest.warns(RuntimeWarning, match="corrupt"):
            assert registry.resolve("codex", scope, "a") is None

        corrupt = registry.path.with_name(registry.path.name + ".corrupt")
        assert corrupt.read_text(encoding="utf-8") == "{ this is not json"
        assert not registry.path.exists()

        registry.bind("codex", scope, "a", "thr-1")
        assert registry.resolve("codex", scope, "a") == "thr-1"
        assert corrupt.exists()

    def test_a_second_corrupt_file_gets_a_timestamped_name(self, registry, scope):
        corrupt = registry.path.with_name(registry.path.name + ".corrupt")
        corrupt.write_text("first", encoding="utf-8")
        registry.path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, wrong shape

        with pytest.warns(RuntimeWarning):
            registry.bind("codex", scope, "a", "thr-1")

        assert corrupt.read_text(encoding="utf-8") == "first"
        extras = [p for p in registry.path.parent.iterdir() if ".corrupt-" in p.name]
        assert len(extras) == 1
        assert extras[0].read_text(encoding="utf-8") == "[1, 2, 3]"
        assert registry.resolve("codex", scope, "a") == "thr-1"

    def test_an_empty_file_is_just_an_empty_registry(self, registry, scope):
        registry.path.write_text("", encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert registry.resolve("codex", scope, "a") is None
        assert not list(registry.path.parent.glob("*.corrupt*"))


class TestConcurrency:
    def test_threads_binding_different_aliases_lose_nothing(self, tmp_path, scope):
        path = tmp_path / "sessions.json"
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                # A registry object per thread: only the lock FILE and the
                # atomic replace protect them from each other.
                reg = SessionRegistry(path)
                for j in range(5):
                    reg.bind("codex", scope, f"a{i}-{j}", f"id-{i}-{j}")
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)

        assert not errors
        rows = SessionRegistry(path).entries()
        assert len(rows) == 40
        assert {r["name"]: r["native_id"] for r in rows}["a3-2"] == "id-3-2"
        json.loads(path.read_text(encoding="utf-8"))  # never a torn file
        assert not list(tmp_path.glob(".*.tmp"))
        assert not list(tmp_path.glob("*.lock"))

    def test_threads_rebinding_one_alias_end_with_a_valid_entry(self, registry, scope):
        def worker(i: int) -> None:
            for j in range(10):
                registry.bind("claude", scope, "shared", f"id-{i}-{j}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)

        assert registry.resolve(
            "claude",
            scope,
            "shared",
        ).startswith("id-")
        assert len(registry.entries()) == 1

    def test_a_stale_lock_file_does_not_block_forever(self, registry, scope, monkeypatch):
        lock = registry.path.with_name(registry.path.name + ".lock")
        lock.write_text("999999", encoding="utf-8")
        old = lock.stat().st_mtime - 10_000
        os.utime(lock, (old, old))

        registry.bind("codex", scope, "a", "thr-1")

        assert registry.resolve("codex", scope, "a") == "thr-1"
        assert not lock.exists()


class TestLocation:
    def test_env_var_names_the_file(self, tmp_path, monkeypatch, scope):
        target = tmp_path / "from-env" / "s.json"
        monkeypatch.setenv(SESSIONS_FILE_ENV, str(target))

        reg = SessionRegistry()
        reg.bind("codex", scope, "a", "thr-1")

        assert reg.path == target
        assert target.exists()

    def test_constructor_path_beats_the_env_var(self, tmp_path, monkeypatch, scope):
        monkeypatch.setenv(SESSIONS_FILE_ENV, str(tmp_path / "env.json"))
        reg = SessionRegistry(tmp_path / "arg.json")
        reg.bind("codex", scope, "a", "thr-1")
        assert (tmp_path / "arg.json").exists()
        assert not (tmp_path / "env.json").exists()

    def test_default_location_is_per_user(self, monkeypatch):
        monkeypatch.delenv(SESSIONS_FILE_ENV, raising=False)
        path = SessionRegistry().path
        assert path.name == "sessions.json"
        assert ".lazybridge" in path.parts

    def test_default_registry_accessor_is_overridable(self, registry):
        previous = set_default_session_registry(registry)
        try:
            assert default_session_registry() is registry
        finally:
            set_default_session_registry(previous)
        set_default_session_registry(None)
        assert default_session_registry() is not registry
        set_default_session_registry(previous)


def test_public_exports():
    assert lazybridge.SessionRegistry is SessionRegistry
    from lazybridge.engines import SessionRegistry as FromEngines

    assert FromEngines is SessionRegistry
    assert sessions_mod.__all__
