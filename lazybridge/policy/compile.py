"""Compile effective policies to native engine settings."""

from __future__ import annotations

import json
from typing import Any

from .models import EffectivePolicy
from .resolve import _thaw


def _metadata(policy: EffectivePolicy) -> dict[str, str | int]:
    return {"policy_version": policy.version, "policy_revision": policy.revision, "policy_digest": policy.digest}


def compile_claude(policy: EffectivePolicy) -> dict[str, Any]:
    """Compile isolated Claude settings and a strict MCP allowlist."""
    servers = {
        name: {key: _thaw(value) for key, value in config.items() if key != "tools"}
        for name, config in sorted(policy.mcp_servers.items())
    }
    allow: list[str] = []
    deny: list[str] = []
    for tool_class, names in (
        ("read", ("Read", "Glob", "Grep")),
        ("edit", ("Edit", "MultiEdit", "Write", "NotebookEdit")),
    ):
        (allow if tool_class in policy.tool_classes else deny).extend(names)
    # No command allowlist exists in the grant schema, so no Bash command is admitted.
    deny.append("Bash")
    (allow if "web" in policy.tools else deny).extend(("WebFetch", "WebSearch"))
    allow.extend(
        f"mcp__{name}__{tool}" for name, config in sorted(policy.mcp_servers.items()) for tool in config["tools"]
    )
    reasons = ["The launcher and workspace restrictions have not been verified."]
    if "shell" in policy.tool_classes:
        reasons.append("No scoped shell commands are specified in the grant.")
    if policy.network_routes:
        reasons.append("Network route restrictions are not enforced by these settings.")
    return {
        **_metadata(policy),
        "eligible": False,
        "ineligibility_reasons": reasons,
        "launch_args": ["--setting-sources", "", "--strict-mcp-config"],
        "settings": {"permissions": {"allow": sorted(allow), "deny": sorted(deny)}},
        "mcp_config": {"mcpServers": servers},
    }


def compile_codex(policy: EffectivePolicy) -> dict[str, Any]:
    """Generate a session CODEX_HOME config; admission requires launch verification."""
    writable = "edit" in policy.tool_classes
    overrides: dict[str, Any] = {
        "sandbox_mode": "workspace-write" if writable else "read-only",
        "approval_policy": "never",
        "web_search": "live" if policy.network == "on" and "web" in policy.tools else "disabled",
        "sandbox_workspace_write": {
            "network_access": policy.network == "on",
            "writable_roots": [str(policy.folder), str(policy.scratch_folder)] if writable else [],
        },
        "mcp_servers": {
            name: (
                {
                    **{
                        key: _thaw(value)
                        for key, value in policy.mcp_servers[name].items()
                        if key not in {"tools", "enabled", "enabled_tools", "disabled_tools"}
                    },
                    "enabled": True,
                    "enabled_tools": list(policy.mcp_servers[name]["tools"]),
                }
                if name in policy.mcp_servers
                else {"enabled": False}
            )
            for name in policy.mcp_inventory
        },
    }
    reasons = ["The launcher has not verified the isolated CODEX_HOME and excluded project configuration."]
    if "shell" not in policy.tool_classes:
        overrides["features"] = {"shell_tool": False}
        reasons.append("Codex shell denial has not been verified for the target runtime.")
    else:
        reasons.append("Scoped shell commands are not enforced by this configuration.")
    if policy.mcp_servers:
        reasons.append("MCP tool filtering and tool-host confinement require native verification.")
    if policy.network_routes:
        reasons.append("Network route restrictions are not enforced by this configuration.")

    return {
        **_metadata(policy),
        "eligible": False,
        "ineligibility_reasons": reasons,
        "codex_home_config_toml": "".join(f"{key} = {_toml_value(value)}\n" for key, value in overrides.items()),
        "config_overrides": overrides,
    }


def _toml_value(value: Any) -> str:
    """Encode the JSON-shaped values used by Codex config, including quoted keys."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_value(key)} = {_toml_value(item)}" for key, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(map(_toml_value, value)) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False).replace("\x7f", r"\u007f")
