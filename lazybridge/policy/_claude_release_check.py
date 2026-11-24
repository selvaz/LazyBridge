"""R1.3 canaries for the local Claude Agent SDK path."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from lazybridge.engines.claude_code.engine import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient
from lazybridge.policy import PolicyDenied, ProfileDocument, TaskGrant, policy_scope, resolve

from ._claude_probe_model import ProbeModel
from .claude_admission import REQUIRED_PROBES


class _FixtureClient(AgentSdkClient):
    """Run the compiled candidate against disposable fixtures before admission."""

    def _require_policy_evidence(self, options: ClaudeSdkOptions) -> Path | None:
        return None


def _resume_policy_probe(start: dict[str, Any], resumed: dict[str, Any]) -> dict[str, Any]:
    """Report resumed enforcement failures without hiding pre-existing gaps."""
    failed = {name for name, probe in resumed.items() if not probe["passed"]}
    missing = start.keys() - resumed.keys()
    prior = sorted(name for name in failed if name in start and not start[name]["passed"])
    regressions = sorted((failed - set(prior)) | missing)
    return {
        "passed": not failed and not missing,
        "detail": f"Pre-existing failures on resume: {', '.join(prior) or 'none'}. "
        f"New or missing failures on resume: {', '.join(regressions) or 'none'}.",
        "probes": resumed,
    }


async def row(
    root: Path,
    executable: str,
    document: ProfileDocument,
    profile: str,
    network: str,
    allowed: str,
    outside: str,
) -> dict[str, Any]:
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    from .release_check import _MCP, _command_args, _endpoint

    workspace, sibling, ambient = (root.resolve() / name for name in ("task", "sibling", "ambient"))
    for folder in (workspace / ".claude", sibling, ambient, workspace / ".scratch"):
        folder.mkdir(parents=True)
    for folder, marker in ((workspace, "task-canary"), (sibling, "sibling-canary")):
        (folder / "read.txt").write_text(marker, encoding="utf-8")
    server = {"command": sys.executable, "args": ["-u", "-c", _MCP, allowed, outside]}
    for path, name in ((ambient / ".claude.json", "ambient_rogue"), (workspace / ".mcp.json", "project_rogue")):
        path.write_text(json.dumps({"mcpServers": {name: server}}), encoding="utf-8")
    hostile = {"permissions": {"allow": ["Bash", "Write", "Read", "WebFetch", "mcp__ambient_rogue"]}}
    for folder in (ambient, workspace / ".claude"):
        (folder / "settings.json").write_text(json.dumps(hostile), encoding="utf-8")
    policy = resolve(
        profile,
        TaskGrant(
            workspace,
            {
                "tool_classes": ["read", "edit", "shell"],
                "tools": ["mcp"],
                "network_routes": [allowed] if network == "on" else [],
                "mcp_servers": {"fixture": {"tools": ["lookup"]}},
                "mcp_inventory": {"fixture": {**server, "tools": ["lookup", "hidden"]}},
            },
            network,
            "local",
        ),
        document,
    )
    engine = ClaudeCodeEngine(cwd=str(workspace), max_turns=4, initialize_timeout=15, stream_idle_timeout=20)
    with policy_scope(policy):
        options = engine._options([], None)

    model_url = ""

    def transport(*, prompt: Any, options: Any) -> Any:
        return SubprocessCLITransport(
            prompt,
            replace(
                options,
                cli_path=executable,
                env={
                    **options.env,
                    "ANTHROPIC_BASE_URL": model_url,
                    "ANTHROPIC_API_KEY": "fixture-key",
                    "ANTHROPIC_AUTH_TOKEN": "",
                    "CLAUDE_CONFIG_DIR": str(ambient),
                    "HOME": str(ambient),
                    "USERPROFILE": str(ambient),
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                    "ENABLE_TOOL_SEARCH": "false",
                },
            ),
        )

    client = _FixtureClient(transport_factory=transport)

    async def attempt(resume: str | None, suffix: str) -> tuple[str, dict[str, Any]]:
        nonlocal model_url
        inside_write = workspace / ".scratch" / f"native-{suffix}.txt"
        outside_write = sibling / f"native-{suffix}.txt"
        shell_file = workspace / ".scratch" / f"shell-{suffix}.txt"
        shell_outside = sibling / f"shell-{suffix}.txt"

        def command(kind: str, value: str) -> str:
            args = _command_args(kind, value, "fixture-ok")
            args[0] = Path(args[0]).as_posix()
            return shlex.join(args)

        calls: list[tuple[str, str, dict[str, Any]]] = [
            ("toolu_in_read", "Read", {"file_path": str(workspace / "read.txt")}),
            ("toolu_out_read", "Read", {"file_path": str(sibling / "read.txt")}),
            ("toolu_in_write", "Write", {"file_path": str(inside_write), "content": "fixture-ok"}),
            ("toolu_out_write", "Write", {"file_path": str(outside_write), "content": "fixture-ok"}),
            (
                "toolu_shell",
                "Bash",
                {
                    "command": "; ".join(
                        [
                            command("write", str(shell_file)),
                            command("write", str(shell_outside)),
                            command("network", allowed),
                            command("network", outside),
                        ]
                    ),
                    "description": "Exercise disposable file and loopback fixtures",
                },
            ),
            ("toolu_mcp", "mcp__fixture__lookup", {}),
            ("toolu_hidden", "mcp__fixture__hidden", {}),
            ("toolu_ambient", "mcp__ambient_rogue__lookup", {}),
            ("toolu_project", "mcp__project_rogue__lookup", {}),
        ]
        model = ProbeModel([(f"{key}_{suffix}", tool, arguments) for key, tool, arguments in calls])
        with _endpoint(model.handler()) as url:
            model_url = url
            result = await asyncio.wait_for(
                client.run("LAZYBRIDGE_RELEASE_FIXTURE", options=replace(options, resume=resume)), timeout=45
            )

        def failure(name: str) -> bool:
            return model.results.get(f"toolu_{name}_{suffix}", {}).get("is_error") is True

        def success(name: str, marker: str) -> bool:
            result = model.results.get(f"toolu_{name}_{suffix}", {})
            return not result.get("is_error") and marker in json.dumps(result.get("content"))

        probes: dict[str, Any] = {}

        def record(name: str, passed: bool, detail: str) -> None:
            probes[name] = {"passed": passed, "detail": detail}

        visible = model.available
        sdk = AgentSdkClient._sdk_options(options)
        expected = set(sdk.allowed_tools)
        required_native = expected & {"Read", "Write"}
        record(
            "isolated_config",
            required_native <= visible <= expected and model.requests >= 2,
            f"Model-facing inventory: {', '.join(sorted(visible)) or 'empty'}. "
            f"Unexpected tools: {', '.join(sorted(visible - expected)) or 'none'}.",
        )
        record(
            "ambient_mcp_absent",
            not any("rogue" in name for name in visible) and failure("ambient") and failure("project"),
            "Both ambient servers must be absent from discovery and reject direct tool calls.",
        )
        mcp_name, hidden_name = "mcp__fixture__lookup", "mcp__fixture__hidden"
        mcp_granted = network == "on"
        record(
            "tool_discovery_and_call",
            hidden_name not in visible
            and failure("hidden")
            and (
                mcp_name in visible and (success("mcp", "[true, false]") or success("mcp", "[true, true]"))
                if mcp_granted
                else mcp_name not in visible and failure("mcp")
            ),
            f"Granted lookup discovered: {mcp_name in visible}. Hidden tool discovered: {hidden_name in visible}. "
            f"Hidden direct call rejected: {failure('hidden')}.",
        )
        record(
            "native_reads",
            (success("in_read", "task-canary") if "read" in policy.tool_classes else failure("in_read"))
            and failure("out_read"),
            f"Scoped read returned canary: {success('in_read', 'task-canary')}. "
            f"Sibling read rejected: {failure('out_read')}.",
        )
        wrote_inside = inside_write.is_file() and inside_write.read_text(encoding="utf-8") == "fixture-ok"
        record(
            "native_writes",
            (wrote_inside if "edit" in policy.tool_classes else failure("in_write") and not inside_write.exists())
            and failure("out_write")
            and not outside_write.exists(),
            f"Scoped write created canary: {wrote_inside}. Sibling write rejected: {failure('out_write')}. "
            f"Sibling write created a file: {outside_write.exists()}.",
        )
        shell_denied = "Bash" not in visible and failure("shell") and not shell_file.exists()
        record(
            "shell_policy",
            shell_denied and "shell" not in policy.tool_classes,
            "Bash needs a positive scoped execution control for a shell grant. "
            f"General shell absent and rejected: {shell_denied}.",
        )
        record(
            "shell_roots",
            shell_denied and not shell_outside.exists() and "shell" not in policy.tool_classes,
            f"Sibling shell marker created: {shell_outside.exists()}. No scoped shell worker is configured.",
        )
        record(
            "shell_egress",
            shell_denied and "shell" not in policy.tool_classes,
            "The shell route is closed only when the native tool is absent and the direct call fails.",
        )
        record(
            "tool_host_egress",
            success("mcp", "[true, false]") if mcp_granted else mcp_name not in visible and failure("mcp"),
            f"Granted endpoint only: {success('mcp', '[true, false]')}. "
            f"Both endpoints reachable: {success('mcp', '[true, true]')}.",
        )
        if not result.session_id:
            raise RuntimeError("The native launch did not return a session ID.")
        return result.session_id, probes

    probes: dict[str, Any] = {}
    try:
        session_id, probes = await attempt(None, "start")
        resumed_id, resumed = await attempt(session_id, "resume")
        probes["resume_effective_policy"] = _resume_policy_probe(probes, resumed)
        probes["thread_reuse"] = {
            "passed": bool(session_id and resumed_id == session_id),
            "detail": "The native resume returned the original session ID."
            if resumed_id == session_id and session_id
            else "The native resume did not return the original session ID.",
        }
        changed = replace(policy, revision=f"{policy.revision}-changed")
        try:
            await client.run(
                "LAZYBRIDGE_RELEASE_FIXTURE", options=replace(options, resume=session_id, effective_policy=changed)
            )
        except PolicyDenied as exc:
            probes["changed_revision_denied"] = {"passed": True, "detail": str(exc)}
    except Exception as exc:
        detail = str(exc).replace(str(root.resolve()), "<fixture>").replace(executable, "<claude>")
        probes["launch"] = {"passed": False, "detail": f"{type(exc).__name__}: {' '.join(detail.split())[:600]}"}
    probes["worker_isolation"] = {
        "passed": False,
        "detail": "The SDK inherits the launcher environment and has no verified confinement for managed settings or worker routes.",
    }
    return {
        "profile": profile,
        "network": network,
        "tool_classes": sorted(policy.tool_classes),
        "status": "green"
        if all(probes.get(name, {}).get("passed") is True for name in REQUIRED_PROBES)
        and all(probe["passed"] for probe in probes.values())
        else "red",
        "probes": {
            **{name: {"passed": False, "detail": "Required probe did not run."} for name in REQUIRED_PROBES},
            **probes,
        },
    }
