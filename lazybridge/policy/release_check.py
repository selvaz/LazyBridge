"""Opt-in local Codex release check. Run with python -m lazybridge.policy.release_check."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from lazybridge.engines.codex.app_server import CodexAppServerClient, CodexRequestRejected, codex_executable
from lazybridge.engines.codex.policy_launch import PolicyLaunch, runtime_revision
from lazybridge.policy import (
    PolicyDenied,
    ProfileDocument,
    TaskGrant,
    compile_codex,
    load_profiles,
    resolve,
)

from ._codex_probe_model import CanaryHandler, ProbeModel
from .codex_admission import LAUNCH_PATH, NETWORKS, PROFILES, REQUIRED_PROBES
from .compile import _toml_value

_MCP = """
import json, sys, urllib.request
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg["method"]
    if method == "initialize":
        result = {"protocolVersion": msg["params"]["protocolVersion"],
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "release-fixture", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": name, "description": name, "inputSchema": {"type": "object"}}
                            for name in ("lookup", "hidden")]}
    elif method == "tools/call":
        observed = []
        for endpoint in sys.argv[1:]:
            try:
                observed.append(urllib.request.urlopen(endpoint, timeout=2).read().decode() == "fixture-ok")
            except OSError:
                observed.append(False)
        result = {"content": [{"type": "text", "text": json.dumps(observed)}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
"""

_COMMAND = """
from pathlib import Path
from urllib.request import urlopen
import sys
kind, value = sys.argv[1:]
if kind == "network":
    try:
        with urlopen(value, timeout=3) as response:
            print(response.read().decode())
    except OSError as exc:
        raise SystemExit(str(exc)) from exc
elif kind == "read":
    print(Path(value).read_text(encoding="utf-8"))
else:
    Path(value).write_text("task-canary" if folder == workspace else "sibling-canary", encoding="utf-8")
"""


@contextmanager
def _endpoint(handler: type[BaseHTTPRequestHandler] = CanaryHandler) -> Iterator[str]:
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(url, timeout=3) as response:
                if response.read() != b"fixture-ok":
                    raise OSError("endpoint did not return the canary")
            yield url
        finally:
            server.shutdown()
            thread.join()


async def _row(
    root: Path, executable: str, document: ProfileDocument, profile: str, network: str, allowed: str, outside: str
) -> dict[str, Any]:
    root = root.resolve()
    workspace, sibling, ambient = (root / name for name in ("task", "sibling", "ambient"))
    for folder in (workspace / ".scratch", sibling, ambient, workspace / ".codex"):
        folder.mkdir(parents=True, exist_ok=True)
    for folder in (workspace, sibling):
        (folder / "read.txt").write_text("task-canary" if folder == workspace else "sibling-canary", encoding="utf-8")
    (workspace / ".git").mkdir()
    (workspace / ".git" / "HEAD").write_text("ref: refs/heads/release-fixture\n", encoding="utf-8")
    hostile = (
        'web_search = "live"\napproval_policy = "on-request"\n'
        "[features]\nshell_tool = true\n[sandbox_workspace_write]\nnetwork_access = true\n"
    )
    for folder, server in ((ambient, "unknown_ambient"), (workspace / ".codex", "unknown_project")):
        (folder / "config.toml").write_text(
            hostile + f'[mcp_servers.{server}]\ncommand = "invalid-fixture-command"\n', encoding="utf-8"
        )
    grant = TaskGrant(
        workspace,
        {
            "tool_classes": ["read", "edit", "shell"],
            "tools": ["mcp", "web"],
            "network_routes": [allowed] if network == "on" else [],
            "mcp_servers": {"fixture": {"tools": ["lookup"]}},
            "mcp_inventory": {
                "fixture": {
                    "command": sys.executable,
                    "args": ["-u", "-c", _MCP, allowed, outside],
                    "tools": ["lookup", "hidden"],
                }
            },
        },
        network,
        "local",
    )
    policy = resolve(profile, grant, document)
    compiled = compile_codex(policy)
    client = CodexAppServerClient(policy_state_dir=root / "state")
    probes: dict[str, dict[str, Any]] = {}

    in_file = workspace / ".scratch" / "native-write.txt"
    out_file = sibling / "native-write.txt"
    shell_marker = workspace / "model-shell.txt"
    if sys.platform == "win32":
        shell = "Set-Content -LiteralPath '" + str(shell_marker).replace("'", "''") + "' -Value fixture-ok"
    else:
        shell = f"printf fixture-ok > {shlex.quote(str(shell_marker))}"
    model = ProbeModel(
        [
            ("read_file", {"path": str(workspace / "read.txt")}),
            ("read_file", {"path": str(sibling / "read.txt")}),
            ("apply_patch", f"*** Begin Patch\n*** Add File: {in_file.as_posix()}\n+fixture-ok\n*** End Patch"),
            ("apply_patch", f"*** Begin Patch\n*** Add File: {out_file.as_posix()}\n+fixture-ok\n*** End Patch"),
            ("shell_command", {"command": shell}),
            ("exec_command", {"cmd": shell}),
        ]
    )

    def record(name: str, passed: bool, detail: str) -> None:
        probes[name] = {"passed": passed, "detail": detail}

    def diagnostic(exc: BaseException) -> str:
        detail = str(exc)
        for path, label in ((str(root), "<fixture>"), (executable, "<codex>")):
            detail = detail.replace(path, label).replace(path.replace("\\", "\\\\"), label)
        return f"{type(exc).__name__}: {' '.join(detail.split())[:800]}"

    async def on_tool_call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("This fixture exercises native MCP.")

    async def inspect_thread(
        request: Callable[[str, dict[str, Any]], Awaitable[Any]],
        thread: dict[str, Any],
        observations: dict[str, dict[str, Any]],
    ) -> None:
        def observe(name: str, passed: bool, detail: str) -> None:
            observations[name] = {"passed": passed, "detail": detail}

        thread_id = thread["thread"]["id"]
        config = await request("config/read", {"cwd": str(workspace), "includeLayers": True})
        layers = config.get("layers") or []
        project_layers = [layer for layer in layers if layer.get("name", {}).get("type") == "project"]
        changing = launch.config_mismatches(config) + launch.thread_mismatches(thread)
        observe(
            "isolated_config",
            not changing and all(layer.get("disabledReason") for layer in project_layers),
            f"Effective settings differing or unavailable: {', '.join(changing) or 'none'}. "
            f"Disabled project layers: {sum(bool(layer.get('disabledReason')) for layer in project_layers)}. "
            f"Reported thread sandbox: {thread.get('sandbox', {}).get('type')}.",
        )

        async def mcp_call(server: str, tool: str) -> Any:
            return await request(
                "mcpServer/tool/call", {"threadId": thread_id, "server": server, "tool": tool, "arguments": {}}
            )

        async def denied(server: str, tool: str) -> bool:
            try:
                return (await mcp_call(server, tool)).get("isError") is True
            except CodexRequestRejected:
                return True

        try:
            status = await request("mcpServerStatus/list", {"threadId": thread_id})
            inventory = {entry["name"]: sorted(entry["tools"]) for entry in status["data"]}
            hidden = {"unknown_ambient", "unknown_project"}
            absent = not (hidden & inventory.keys())
            denied_ambient = all([await denied(name, "lookup") for name in sorted(hidden)])
            observe("ambient_mcp_absent", absent and denied_ambient, f"Discovered inventory: {inventory}.")
            allowed_result = await mcp_call("fixture", "lookup") if network == "on" else None
            expected_tools = {"fixture": ["lookup"]} if network == "on" else {"fixture": []}
            hidden_denied = await denied("fixture", "hidden")
            granted_ok = (
                not allowed_result.get("isError") if allowed_result is not None else await denied("fixture", "lookup")
            )
            exact = (inventory == expected_tools or (network == "off" and not inventory)) and status.get(
                "nextCursor"
            ) is None
            observe(
                "tool_discovery_and_call",
                exact and hidden_denied and granted_ok,
                f"Inventory: {inventory}. Hidden call denied: {hidden_denied}. Granted control: {granted_ok}.",
            )
            if allowed_result is None:
                observe(
                    "tool_host_egress",
                    exact and granted_ok and denied_ambient,
                    "No configured or ambient tool host was callable with networking off.",
                )
            else:
                content = allowed_result.get("content") or []
                observed = json.loads(content[0]["text"]) if content else None
                observe(
                    "tool_host_egress",
                    observed == [True, False],
                    f"Tool-host endpoint reachability [declared, undeclared]: {observed}.",
                )
        except (CodexRequestRejected, KeyError, TypeError, ValueError) as exc:
            for name in ("ambient_mcp_absent", "tool_discovery_and_call", "tool_host_egress"):
                if name not in observations:
                    observe(name, False, diagnostic(exc))

        async def command(kind: str, value: str) -> tuple[bool, str]:
            try:
                response = await request(
                    "command/exec",
                    {
                        "command": [sys.executable, "-c", _COMMAND, kind, value],
                        "cwd": str(workspace),
                        "sandboxPolicy": thread["sandbox"],
                        "timeoutMs": 7000,
                    },
                )
                return response["exitCode"] == 0, (response.get("stdout") or response.get("stderr") or "")[:160]
            except CodexRequestRejected as exc:
                return False, diagnostic(exc)

        inside_read, control = await command("read", str(workspace / "read.txt"))
        outside_read, _ = await command("read", str(sibling / "read.txt"))
        inside_write, _ = await command("write", str(workspace / ".scratch" / "shell-write.txt"))
        outside_write, _ = await command("write", str(sibling / "shell-write.txt"))
        control_ok = inside_read and control.strip() == "task-canary"
        expected_write = profile != "read"
        observe(
            "shell_roots",
            control_ok and not outside_read and inside_write == expected_write and not outside_write,
            f"Reads [task, sibling]: {[inside_read, outside_read]}. Writes [scratch, sibling]: "
            f"{[inside_write, outside_write]}. Control output: {' '.join(control.split())[:180]}",
        )
        allowed_ok, allowed_output = await command("network", allowed)
        outside_ok, outside_output = await command("network", outside)
        expected_egress = network == "on" and "shell" in policy.tool_classes
        observe(
            "shell_egress",
            control_ok
            and allowed_ok == expected_egress
            and (not allowed_ok or allowed_output.strip() == "fixture-ok")
            and not outside_ok,
            f"Shell endpoint reachability [declared, undeclared]: {[allowed_ok, outside_ok]}. "
            f"Command control: {control_ok}. Undeclared output: {' '.join(outside_output.split())[:160]}",
        )

    async def probe(request: Any, thread: dict[str, Any]) -> bool:
        await inspect_thread(request, thread, probes)
        return True

    kwargs: dict[str, Any] = {
        "prompt": "Run the scripted release fixture.",
        "model": "gpt-5",
        "dynamic_tools": [],
        "on_tool_call": on_tool_call,
        "developer_instructions": None,
        "on_text": None,
        "attachments": None,
        "effort": None,
        "output_schema": None,
        "ephemeral": False,
        "review_target": None,
        "progress": None,
        "thread_source": "release-check",
    }
    previous = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(ambient)
    try:
        with _endpoint(model.handler()) as model_url:
            model_config = {
                "model_provider": "release-fixture",
                "model_providers": {
                    "release-fixture": {
                        "name": "Release fixture",
                        "base_url": f"{model_url}/v1",
                        "wire_api": "responses",
                        "supports_websockets": False,
                        "requires_openai_auth": False,
                    }
                },
            }
            fixture_config = {
                **compiled,
                "config_overrides": {**compiled["config_overrides"], **model_config},
                "codex_home_config_toml": compiled["codex_home_config_toml"]
                + "".join(f"{key} = {_toml_value(value)}\n" for key, value in model_config.items()),
            }
            launch = PolicyLaunch(policy, fixture_config, executable)
            result = await asyncio.wait_for(
                client._run_policy_launch(launch, {**kwargs, "thread_id": None}, probe=probe), 45
            )
        read_in = "task-canary" in str(model.outputs.get("call-0"))
        read_out = "sibling-canary" in str(model.outputs.get("call-1"))
        reads_returned = {"call-0", "call-1"} <= model.outputs.keys()
        record(
            "native_reads",
            reads_returned and read_in and not read_out,
            f"Model read canaries [task, sibling]: {[read_in, read_out]}. "
            f"read_file advertised: {'read_file' in model.available}. Both calls returned: {reads_returned}.",
        )
        write_in, write_out = in_file.is_file(), out_file.is_file()
        writes_returned = {"call-2", "call-3"} <= model.outputs.keys()
        record(
            "native_writes",
            writes_returned and write_in == (profile != "read") and not write_out,
            f"Model apply_patch created [scratch, sibling]: {[write_in, write_out]}. "
            f"apply_patch advertised: {'apply_patch' in model.available}. Both calls returned: {writes_returned}.",
        )
        shell_tools = sorted(
            {"shell", "shell_command", "exec_command", "write_stdin", "local_shell"} & model.available.keys()
        )
        commands_returned = {"call-4", "call-5"} <= model.outputs.keys()
        record(
            "shell_policy",
            profile != "edit-and-run" and not shell_tools and commands_returned and not shell_marker.exists(),
            f"Model shell inventory: {shell_tools}. Both direct calls returned: {commands_returned}. "
            f"Shell canary was created: {shell_marker.exists()}. "
            + ("A scoped command allowlist is unavailable." if profile == "edit-and-run" else ""),
        )

        async def resumed(request: Any, thread: dict[str, Any]) -> bool:
            record(
                "thread_reuse",
                thread["thread"]["id"] == result.thread_id,
                "Native thread/resume returned the bound ID.",
            )
            observations: dict[str, dict[str, Any]] = {}
            try:
                await inspect_thread(request, thread, observations)
            finally:
                checked = ("isolated_config", "ambient_mcp_absent", "tool_discovery_and_call", "shell_egress")
                failed = [name for name in checked if not observations.get(name, {}).get("passed")]
                record(
                    "resume_effective_policy",
                    not failed,
                    f"Repeated configuration, ambient inventory, direct calls and shell egress on resume. "
                    f"Failed or missing: {', '.join(failed) or 'none'}.",
                )
            return False

        resume_kwargs = {**kwargs, "thread_id": result.thread_id}
        changed = resolve(profile, grant, replace(document, revision=document.revision + ".changed"))
        try:
            await client._run_policy_launch(
                PolicyLaunch(changed, compile_codex(changed), executable), resume_kwargs, probe=resumed
            )
        except PolicyDenied:
            record("changed_revision_denied", True, "The requested revision differed from the stored thread binding.")
        else:
            record("changed_revision_denied", False, "A thread was resumed under a different policy revision.")
        try:
            await asyncio.wait_for(client._run_policy_launch(launch, resume_kwargs, probe=resumed), 15)
        except Exception as exc:
            if "thread_reuse" in probes:
                record("resume_effective_policy", False, diagnostic(exc))
            else:
                record("thread_reuse", False, diagnostic(exc))
    except Exception as exc:
        record("launch", False, diagnostic(exc))
    finally:
        if previous is None:
            del os.environ["CODEX_HOME"]
        else:
            os.environ["CODEX_HOME"] = previous
    for name in REQUIRED_PROBES:
        probes.setdefault(name, {"passed": False, "detail": "Required probe did not run."})
    return {
        "profile": profile,
        "network": network,
        "tool_classes": sorted(policy.tool_classes),
        "status": "green" if all(probe_result["passed"] for probe_result in probes.values()) else "red",
        "probes": probes,
    }


async def run_check(document: ProfileDocument, fixture_parent: Path | None = None) -> dict[str, Any]:
    """Run all six rows against the local native CLI in disposable fixtures."""
    checked_at = datetime.now(UTC).isoformat()
    version, revision = "unavailable", None
    matrix = []
    try:
        executable = codex_executable()
        version = subprocess.check_output([executable, "--version"], text=True, timeout=10).strip()
        revision = runtime_revision(executable)
        with (
            tempfile.TemporaryDirectory(prefix="codex-release-", dir=fixture_parent) as temp,
            _endpoint() as allowed,
            _endpoint() as outside,
        ):
            for profile in PROFILES:
                for network in NETWORKS:
                    matrix.append(
                        await _row(
                            Path(temp) / f"{profile}-{network}",
                            executable,
                            document,
                            profile,
                            network,
                            allowed,
                            outside,
                        )
                    )
    except (OSError, subprocess.SubprocessError) as exc:
        completed = {(row["profile"], row["network"]) for row in matrix}
        matrix.extend(
            {
                "profile": profile,
                "network": network,
                "status": "red",
                "probes": {
                    name: {"passed": False, "detail": f"Required probe unavailable: {type(exc).__name__}: {exc}"}
                    for name in REQUIRED_PROBES
                },
            }
            for profile in PROFILES
            for network in NETWORKS
            if (profile, network) not in completed
        )
    return {
        "schema": 1,
        "checked_at": checked_at,
        "launch_path": LAUNCH_PATH,
        "policy_version": document.version,
        "policy_revision": document.revision,
        "runtime_version": version,
        "runtime_revision": revision,
        "platform": platform.platform(),
        "matrix": matrix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, help="Canonical profile TOML (defaults to bundled example).")
    parser.add_argument("--output", type=Path, help="Write the JSON release report to this file.")
    parser.add_argument("--fixture-parent", type=Path, help="Parent directory for disposable probe fixtures.")
    args = parser.parse_args()
    document = load_profiles(args.profiles) if args.profiles else load_profiles()
    report = asyncio.run(run_check(document, args.fixture_parent))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if all(row["status"] == "green" for row in report["matrix"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
