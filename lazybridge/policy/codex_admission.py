"""Evidence required to admit a governed local Codex launch."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .models import EffectivePolicy

LAUNCH_PATH = "CodexAppServerClient/stdio"
PROFILES = ("read", "edit", "edit-and-run")
NETWORKS = ("off", "on")
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
)


def admission_id(
    policy: EffectivePolicy, release_report: Path | str | None, runtime_revision: str | None
) -> str | None:
    """Return the identity of a complete, matching green report row."""
    if release_report is None or not runtime_revision or policy.placement != "local":
        return None
    try:
        report = json.loads(Path(release_report).read_text(encoding="utf-8"))
        if (
            report["schema"] != 1
            or report["launch_path"] != LAUNCH_PATH
            or report["runtime_revision"] != runtime_revision
            or report["policy_version"] != policy.version
            or report["policy_revision"] != policy.revision
            or datetime.fromisoformat(report["checked_at"]).utcoffset() is None
        ):
            return None
        rows = [row for row in report["matrix"] if (row["profile"], row["network"]) == (policy.profile, policy.network)]
        if len(rows) != 1 or rows[0]["status"] != "green" or rows[0]["tool_classes"] != sorted(policy.tool_classes):
            return None
        probes = rows[0]["probes"]
        if not all(probes.get(name, {}).get("passed") is True for name in REQUIRED_PROBES):
            return None
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return None
    return hashlib.sha256(json.dumps({**report, "matrix": rows}, sort_keys=True).encode()).hexdigest()
