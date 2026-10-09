from __future__ import annotations

import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from lazybridge.engines.codex import app_server


@pytest.fixture
def app_installs(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.setattr(app_server.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    versions = {}
    calls = Counter()

    def probe(executable):
        calls[executable] += 1
        return versions[executable]

    monkeypatch.setattr(app_server, "_probe_codex_version", probe)
    app_server._select_app_codex.cache_clear()

    def install(directory, version="codex-cli 0.160.0", mtime=1, name="codex.exe", root=None):
        root = root or tmp_path / "local" / "OpenAI" / "Codex" / "bin"
        executable = root / directory / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
        os.utime(executable, (mtime, mtime))
        versions[str(executable)] = version
        return str(executable)

    yield install, calls
    app_server._select_app_codex.cache_clear()


def test_only_exact_cli_files_directly_under_hash_directories(app_installs):
    install, calls = app_installs
    cli = install("stable")
    for name in (
        "codex-code-mode-host.exe",
        "codex-command-runner.exe",
        "codex-windows-sandbox-setup.exe",
        "codex-helper",
        "codex.exe.bak",
    ):
        install("stable", mtime=100, name=name)
    install("stable/nested", mtime=100)
    install(".", mtime=100)
    root = Path(cli).parent.parent
    (root / "directory" / "codex.exe").mkdir(parents=True)

    assert app_server.codex_executable() == cli
    assert calls == {cli: 1}


def test_stable_beats_alpha_with_higher_version_and_mtime(app_installs):
    install, _ = app_installs
    stable = install("stable", "codex-cli 0.160.0", mtime=1)
    install("alpha", "codex-cli 0.162.0-alpha.2", mtime=100)

    assert app_server.codex_executable() == stable


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("0.9.0", "0.10.0"),
        ("0.160.0", "0.160.1"),
        ("0.999.999", "1.0.0"),
        ("0.162.0-alpha.2", "0.162.0-alpha.10"),
        ("0.162.0-alpha.99", "0.163.0-alpha.1"),
        ("0.162.0-alpha", "0.162.0-alpha.1"),
        ("0.162.0-alpha.1", "0.162.0-beta.1"),
        ("0.162.0-alpha.10", "0.162.0-alpha.beta"),
    ],
)
def test_versions_use_numeric_and_semver_prerelease_order(app_installs, lower, higher):
    install, _ = app_installs
    install("lower", f"codex-cli {lower}", mtime=100)
    expected = install("higher", f"codex-cli {higher}", mtime=1)

    assert app_server.codex_executable() == expected


@pytest.mark.parametrize("invalid", [None, "not a version", "codex-cli 0.162.0-alpha.02"])
def test_unparseable_or_failed_probe_ranks_below_parsed_prerelease(app_installs, invalid):
    install, _ = app_installs
    install("invalid", invalid, mtime=100)
    parsed = install("parsed", "codex-cli 0.1.0-alpha.1", mtime=1)

    assert app_server.codex_executable() == parsed


def test_failed_probes_fall_back_to_mtime(app_installs):
    install, _ = app_installs
    install("older", None, mtime=1)
    newest = install("newer", "unknown", mtime=100)

    assert app_server.codex_executable() == newest


def test_equal_versions_use_mtime_ignoring_build_metadata(app_installs):
    install, _ = app_installs
    install("older", "codex-cli 0.160.0+build.z", mtime=1)
    newest = install("newer", "codex-cli 0.160.0+build.a\n", mtime=100)

    assert app_server.codex_executable() == newest


def test_extensionless_cli_in_home_app_root(app_installs, tmp_path):
    install, _ = app_installs
    cli = install("hash", name="codex", root=tmp_path / "home" / ".local" / "share" / "OpenAI" / "Codex" / "bin")

    assert app_server.codex_executable() == cli


def test_codex_bin_short_circuits_path_and_probes(app_installs, monkeypatch):
    install, calls = app_installs
    install("stable")
    monkeypatch.setenv("CODEX_BIN", "custom/codex")

    def unexpected_path_lookup(name):
        pytest.fail("CODEX_BIN must short-circuit PATH lookup")

    monkeypatch.setattr(app_server.shutil, "which", unexpected_path_lookup)
    assert app_server.codex_executable() == "custom/codex"
    monkeypatch.setenv("CODEX_BIN", "another/codex")
    assert app_server.codex_executable() == "another/codex"
    assert not calls


def test_path_short_circuits_probes_and_is_not_cached(app_installs, monkeypatch):
    install, calls = app_installs
    install("stable")
    monkeypatch.setattr(app_server.shutil, "which", lambda name: "path/codex")
    assert app_server.codex_executable() == "path/codex"
    monkeypatch.setattr(app_server.shutil, "which", lambda name: "new-path/codex")
    assert app_server.codex_executable() == "new-path/codex"
    assert not calls


def test_selection_is_cached_with_one_probe_per_executable(app_installs):
    install, calls = app_installs
    stable = install("stable")
    alpha = install("alpha", "codex-cli 0.162.0-alpha.2")

    for _ in range(3):
        assert app_server.codex_executable() == stable
    assert calls == {stable: 1, alpha: 1}


def test_cache_is_invalidated_when_mtime_or_candidates_change(app_installs):
    install, calls = app_installs
    first = install("first", mtime=1)
    second = install("second", mtime=2)
    assert app_server.codex_executable() == second
    os.utime(first, (3, 3))
    assert app_server.codex_executable() == first
    newest = install("newest", "codex-cli 0.161.0")
    assert app_server.codex_executable() == newest
    assert calls == {first: 3, second: 3, newest: 1}


def test_no_app_cli_raises(app_installs):
    _, calls = app_installs
    with pytest.raises(FileNotFoundError, match="codex CLI not found"):
        app_server.codex_executable()
    assert not calls


def test_probe_runs_version_with_short_timeout(monkeypatch):
    def run(command, **kwargs):
        assert command == ["app/codex.exe", "--version"]
        assert kwargs == {
            "capture_output": True,
            "check": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": 10,
        }
        return subprocess.CompletedProcess(command, 0, stdout="codex-cli 0.160.0\n")

    monkeypatch.setattr(app_server.subprocess, "run", run)
    assert app_server._probe_codex_version("app/codex.exe") == "codex-cli 0.160.0\n"


@pytest.mark.parametrize(
    "error",
    [OSError("cannot execute"), subprocess.CalledProcessError(1, "codex"), subprocess.TimeoutExpired("codex", 10)],
)
def test_probe_failures_return_none(monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(app_server.subprocess, "run", run)
    assert app_server._probe_codex_version("app/codex.exe") is None
