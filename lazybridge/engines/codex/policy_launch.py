"""Isolated configuration, credentials and thread storage for governed Codex."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from shutil import rmtree
from typing import Any
from uuid import UUID

from lazybridge.policy import EffectivePolicy, PolicyDenied, compile_codex

_ENVIRONMENT = frozenset({"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "LANG", "LC_ALL"})


def runtime_revision(executable: str) -> str:
    binary = Path(executable).resolve(strict=True)
    stat = binary.stat()
    return _runtime_revision(binary, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _runtime_revision(binary: Path, size: int, mtime_ns: int) -> str:
    with binary.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    implementation = Path(__file__).parent
    policy = implementation.parent.parent / "policy"
    digest = hashlib.sha256(repr((fingerprint, platform.system(), platform.release(), platform.version())).encode())
    for path in (
        implementation / "app_server.py",
        implementation / "engine.py",
        implementation / "policy_launch.py",
        policy / "compile.py",
        policy / "codex_admission.py",
        policy / "release_check.py",
        policy / "_codex_probe_model.py",
    ):
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _thread_record(state: Path, thread_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", thread_id):
        raise PolicyDenied("codex", "Invalid governed Codex thread ID.")
    return state / f"{thread_id}.json"


def _config_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _config_contains(actual[key], value) for key, value in expected.items()
        )
    return bool(actual == expected)


class PolicyLaunch:
    """One compiled launch and the state it is authorized to access."""

    def __init__(self, policy: EffectivePolicy, compiled: dict[str, Any], executable: str) -> None:
        self.policy = policy
        self.compiled = compiled
        self.executable = executable
        self.config: dict[str, Any] = compiled["config_overrides"]

    @classmethod
    def admit(cls, policy: EffectivePolicy, executable: str, report: Path | None) -> PolicyLaunch:
        binary = Path(executable).resolve(strict=True)
        if binary.is_relative_to(policy.folder):
            raise PolicyDenied("codex", "The Codex executable must be outside the assigned folder.")
        if report is not None and report.resolve().is_relative_to(policy.folder):
            raise PolicyDenied("codex", "The release report must be outside the assigned folder.")
        compiled = compile_codex(policy, release_report=report, runtime_revision=runtime_revision(str(binary)))
        if not compiled["eligible"]:
            raise PolicyDenied("codex", " ".join(compiled["ineligibility_reasons"]))
        return cls(policy, compiled, str(binary))

    @contextmanager
    def home(
        self,
        thread_id: str | None,
        durable: bool,
        state: Path | None,
        api_key: str | None,
        credential_id: UUID | None = None,
    ) -> Iterator[tuple[Path, dict[str, str]]]:
        if durable and api_key is not None and not isinstance(credential_id, UUID):
            raise PolicyDenied("codex", "Durable API-key sessions require an executor-assigned credential UUID.")
        binding = {
            "admission_id": self.compiled["admission_id"],
            "policy_digest": self.policy.digest,
            "credential_id": str(credential_id) if api_key is not None and credential_id is not None else None,
        }
        if durable:
            if state is None:
                raise PolicyDenied("codex", "Governed Codex threads require a trusted policy_state_dir.")
            state = state.resolve()
            if state == self.policy.folder or self.policy.folder in state.parents:
                raise PolicyDenied("codex", "Codex thread state must be outside the assigned folder.")
        if thread_id is not None:
            assert state is not None
            try:
                record = json.loads(_thread_record(state, thread_id).read_text(encoding="utf-8"))
                home_name = record["home"]
                if not re.fullmatch(r"codex-[A-Za-z0-9_]+", home_name) or record != {**binding, "home": home_name}:
                    raise ValueError("binding differs")
                home = state / home_name
                if home.resolve() != home or not home.is_dir():
                    raise ValueError("home differs")
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise PolicyDenied("codex", "Codex thread does not match the current admission.") from exc
        else:
            if durable:
                assert state is not None
                state.mkdir(parents=True, exist_ok=True)
            home = Path(tempfile.mkdtemp(prefix="codex-", dir=state if durable else None)).resolve()
        self.home_path = home
        self.binding = binding
        self.state = state if durable else None
        self.bound = thread_id is not None
        try:
            config = self.compiled["codex_home_config_toml"]
            path = home / "config.toml"
            if thread_id is None:
                path.write_text(config, encoding="utf-8")
            elif path.is_symlink() or path.read_text(encoding="utf-8") != config:
                raise PolicyDenied("codex", "The stored Codex config differs from the current admission.")
            env = {name: value for name, value in os.environ.items() if name.upper() in _ENVIRONMENT}
            env["CODEX_HOME"] = str(home)
            if api_key is not None:
                env["OPENAI_API_KEY"] = api_key
            # Codex creates helper shims and SQLite temporary files here.
            # Its helper installer refuses a temp directory inside CODEX_HOME.
            with tempfile.TemporaryDirectory(prefix="codex-tmp-", dir=home.parent) as temp:
                for name in ("TEMP", "TMP", "TMPDIR"):
                    env[name] = temp
                yield home, env
        finally:
            if not self.bound:
                rmtree(home)

    def bind_thread(self, thread_id: str) -> None:
        if self.state is None or self.bound:
            return
        record = _thread_record(self.state, thread_id)
        data = json.dumps({**self.binding, "home": self.home_path.name})
        with record.open("x", encoding="utf-8") as stream:
            stream.write(data)
        self.bound = True

    def config_mismatches(self, snapshot: dict[str, Any]) -> list[str]:
        expected = self.compiled["config_overrides"]
        actual = snapshot.get("config") or {}
        mismatches = [key for key, value in expected.items() if not _config_contains(actual, {key: value})]
        if set(actual.get("mcp_servers", {})) != set(expected["mcp_servers"]):
            mismatches.append("MCP inventory")
        sources = []
        for layer in snapshot.get("layers") or ():
            source = layer["name"]
            if layer.get("disabledReason"):
                continue
            if source["type"] == "user":
                sources.append(Path(source["file"]).resolve())
            elif source["type"] != "packagedDefaults" and (layer.get("config") or source["type"] == "project"):
                mismatches.append(f"enabled {source['type']} config layer")
        if sources != [self.home_path / "config.toml"]:
            mismatches.append("isolated user config layer")
        return mismatches

    def thread_mismatches(self, thread: dict[str, Any]) -> list[str]:
        expected = self.config
        sandbox = thread.get("sandbox") or {}
        writable = expected["sandbox_mode"] == "workspace-write"
        mismatches = []
        if thread.get("approvalPolicy") != "never":
            mismatches.append("thread approval policy")
        if not thread.get("cwd") or Path(thread["cwd"]).resolve() != self.policy.folder:
            mismatches.append("thread cwd")
        if sandbox.get("type") != ("workspaceWrite" if writable else "readOnly"):
            mismatches.append("thread sandbox mode")
        if writable:
            if sandbox.get("networkAccess") is not expected["sandbox_workspace_write"]["network_access"]:
                mismatches.append("thread network")
            roots = {Path(path).resolve() for path in sandbox.get("writableRoots", ())}
            if roots != {self.policy.folder, self.policy.scratch_folder}:
                mismatches.append("thread writable roots")
        return mismatches
