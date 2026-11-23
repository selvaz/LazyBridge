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
    allow = []
    deny = []
    for tool_class, names in (("read", ("Read", "Glob", "Grep")), ("edit", ("Edit", "Write"))):
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
    overrides = {
        "sandbox_mode": "workspace-write" if writable else "read-only",
        "approval_policy": "never",
        "sandbox_workspace_write": {
            "network_access": policy.network == "on",
            "writable_roots": [str(policy.folder), str(policy.scratch_folder)] if writable else [],
        },
        "tools": {"web_search": policy.network == "on" and "web" in policy.tools},
        "mcp_servers": {name: {"enabled": name in policy.mcp_servers} for name in policy.mcp_inventory},
    }
    reasons = ["The launcher has not verified the isolated CODEX_HOME and excluded project configuration."]
    if "shell" not in policy.tool_classes:
        reasons.append("Codex shell denial is not enforced by this configuration.")
    else:
        reasons.append("Scoped shell commands are not enforced by this configuration.")
    if policy.mcp_servers:
        reasons.append("MCP server definitions and per-tool restrictions are not enforced by this configuration.")
    if policy.network_routes:
        reasons.append("Network route restrictions are not enforced by this configuration.")

    def quote(value: object) -> str:
        return json.dumps(value, ensure_ascii=True)

    lines = [
        f"sandbox_mode = {quote(overrides['sandbox_mode'])}",
        'approval_policy = "never"',
        "[sandbox_workspace_write]",
        f"network_access = {str(policy.network == 'on').lower()}",
        f"writable_roots = {quote(overrides['sandbox_workspace_write']['writable_roots'])}",
        "[tools]",
        f"web_search = {str(overrides['tools']['web_search']).lower()}",
    ]
    for name, config in overrides["mcp_servers"].items():
        lines.extend([f"[mcp_servers.{quote(name)}]", f"enabled = {str(config['enabled']).lower()}"])
    return {
        **_metadata(policy),
        "eligible": False,
        "ineligibility_reasons": reasons,
        "codex_home_config_toml": "\n".join(lines) + "\n",
        "config_overrides": overrides,
    }
