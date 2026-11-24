"""Evidence contract for governed local Claude launches."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from functools import lru_cache
from pathlib import Path

from .codex_admission import admission_id
from .models import EffectivePolicy

LAUNCH_PATH = "AgentSdkClient/stdio"
REQUIRED_PROBES = (
    "isolated_config",
    "ambient_mcp_absent",
    "tool_discovery_and_call",
    "native_reads",
    "native_writes",
    "shell_policy",
    "shell_roots",
    "shell_egress",
    "tool_host_egress",
    "thread_reuse",
    "resume_effective_policy",
    "changed_revision_denied",
    "worker_isolation",
)


def claude_admission_id(policy: EffectivePolicy, report: Path | str | None, revision: str | None) -> str | None:
    return admission_id(policy, report, revision, launch_path=LAUNCH_PATH, required_probes=REQUIRED_PROBES)


def runtime_revision(binary: Path) -> str:
    stat = binary.stat()
    return _runtime_revision(binary, stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _runtime_revision(binary: Path, size: int, mtime_ns: int) -> str:
    with binary.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    sdk_version = importlib.metadata.version("claude-agent-sdk")
    digest = hashlib.sha256(repr((fingerprint, platform.platform(), sdk_version)).encode())
    policy = Path(__file__).parent
    for name in (
        "compile.py",
        "claude_admission.py",
        "codex_admission.py",
        "release_check.py",
        "_claude_probe_model.py",
        "_claude_release_check.py",
    ):
        digest.update((policy / name).read_bytes())
    for name in ("sdk_client.py", "engine.py", "protocol.py"):
        digest.update((policy.parent / "engines" / "claude_code" / name).read_bytes())
    return digest.hexdigest()
