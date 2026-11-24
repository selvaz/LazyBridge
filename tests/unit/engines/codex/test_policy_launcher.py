from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest

from lazybridge import Agent, CodexEngine, Tool
from lazybridge.engines.codex import app_server
from lazybridge.engines.codex.app_server import CodexAppServerClient
from lazybridge.engines.codex.policy_launch import runtime_revision
from lazybridge.policy import TaskGrant, compile_codex, load_profiles, release_check, resolve
from lazybridge.policy.codex_admission import LAUNCH_PATH, REQUIRED_PROBES

THREAD = "01020304-0102-0304-0102-010203040506"
SERVER = """
import json, pathlib, sys, tomllib
def read(method):
    request = json.loads(sys.stdin.readline())
    assert request["method"] == method, request
    return request
def send(**message):
    print(json.dumps(message), flush=True)
def reply(request, result):
    send(id=request["id"], result=result)
reply(read("initialize"), {})
read("initialized")
config = pathlib.Path("config.toml").resolve()
effective = tomllib.loads(config.read_text())
if sys.argv[3] == "drift":
    effective["mcp_servers"]["unknown"] = {"enabled": True}
reply(read("config/read"), {
    "config": effective,
    "layers": [{"name": {"type": "user", "file": str(config)}}],
})
request = json.loads(sys.stdin.readline())
with open(sys.argv[2], "w") as stream:
    json.dump(request, stream)
params = request["params"]
reply(request, {
    "approvalPolicy": params["approvalPolicy"],
    "cwd": params["cwd"],
    "sandbox": {
        "type": "workspaceWrite",
        "networkAccess": False,
        "writableRoots": params["config"]["sandbox_workspace_write"]["writable_roots"],
    },
    "thread": {"id": sys.argv[1]},
})
turn = read("turn/start")
reply(turn, {"turn": {"id": "new"}})
for name in ("allowed", "hidden"):
    send(id=name, method="item/tool/call", params={"tool": name, "arguments": {}})
    result = json.loads(sys.stdin.readline())
    assert result["result"]["success"] == (name == "allowed"), result
send(method="turn/completed", params={"turn": {"id": "new", "status": "completed", "items": [
    {"type": "agentMessage", "text": "checked"}
]}})
"""


def test_release_write_probe_writes_the_supplied_payload(tmp_path):
    target = tmp_path / "scratch write.txt"
    result = subprocess.run(
        release_check._command_args("write", str(target), "write-control"), capture_output=True, timeout=5
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == "write-control"


def report_for(policy, runtime):
    return {
        "schema": 1,
        "launch_path": LAUNCH_PATH,
        "policy_revision": policy.revision,
        "policy_version": policy.version,
        "runtime_revision": runtime,
        "checked_at": "2026-11-23T12:00:00+00:00",
        "matrix": [
            {
                "profile": policy.profile,
                "network": policy.network,
                "tool_classes": sorted(policy.tool_classes),
                "status": "green",
                "probes": {probe: {"passed": True} for probe in REQUIRED_PROBES},
            }
        ],
    }


def test_codex_receipt_requires_complete_matching_evidence(tmp_path):
    report_path = tmp_path / "report.json"
    policy = resolve("read", TaskGrant(tmp_path, {"tool_classes": ["read"]}, "off", "local"))
    report = report_for(policy, "runtime")
    opts = {"release_report": report_path, "runtime_revision": "runtime"}
    assert not compile_codex(policy, **opts)["eligible"]
    report_path.write_text(json.dumps(report))
    assert compile_codex(policy, **opts)["eligible"]
    assert not compile_codex(replace(policy, revision="changed"), **opts)["eligible"]
    assert not compile_codex(replace(policy, network="on"), **opts)["eligible"]
    assert not compile_codex(policy, **{**opts, "runtime_revision": "changed"})["eligible"]
    report["matrix"][0]["probes"].pop("shell_policy")
    report_path.write_text(json.dumps(report))
    assert not compile_codex(policy, **opts)["eligible"]
    report["matrix"][0]["probes"]["shell_policy"] = {"passed": False}
    report_path.write_text(json.dumps(report))
    assert not compile_codex(policy, **opts)["eligible"]


@pytest.mark.asyncio
async def test_governed_config_rpc_tool_filter_and_resume(tmp_path, monkeypatch):
    workspace = tmp_path / "task"
    workspace.mkdir()
    fixture = tmp_path / "fake.py"
    fixture.write_text(SERVER)
    executable = tmp_path / "codex"
    executable.write_bytes(b"pinned runtime")
    monkeypatch.setattr(app_server, "codex_executable", lambda: str(executable))
    monkeypatch.setattr(app_server, "_display_version", lambda _: "fixture")
    monkeypatch.setattr(CodexAppServerClient, "prerelease_warning", lambda _: None)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "ambient"))
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    monkeypatch.setenv("LAZYBRIDGE_CODEX_TRANSPORT", "invalid-with-policy")
    grant = TaskGrant(workspace, {"tool_classes": ["read", "edit"]}, "off", "local")
    policy = resolve("edit", grant)
    report = tmp_path / "report.json"
    revision = runtime_revision(str(executable))
    report.write_text(json.dumps(report_for(policy, revision)))
    client = CodexAppServerClient(policy_release_report=report, policy_state_dir=tmp_path / "state")
    create_subprocess = asyncio.create_subprocess_exec
    launches = []
    drift = False

    async def spawn(*args, **kwargs):
        assert args == (str(executable), "app-server", "--strict-config")
        home = Path(kwargs["env"]["CODEX_HOME"])
        assert kwargs["cwd"] == home
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert home.parent == tmp_path / "state"
        assert {path.name for path in home.iterdir()} == {"config.toml"}
        assert tomllib.loads((home / "config.toml").read_text()) == compile_codex(policy)["config_overrides"]
        capture = tmp_path / f"rpc-{len(launches)}.json"
        launches.append(capture)
        return await create_subprocess(
            sys.executable, str(fixture), THREAD, str(capture), "drift" if drift else "", **kwargs
        )

    monkeypatch.setattr(app_server.asyncio, "create_subprocess_exec", spawn)
    invoked = []

    def allowed() -> str:
        invoked.append("allowed")
        return "allowed"

    def hidden() -> str:
        raise AssertionError("Ungrantable tool was called.")

    tools = [Tool(allowed, policy_grant="read"), Tool(hidden, policy_grant="shell")]
    engine = CodexEngine(client=client, persist_thread=True, max_retries=0)
    agent = Agent(engine, tools=tools, policy=policy)
    result = await agent.run("check")
    assert result.ok, result.error
    assert engine.thread_id == THREAD
    resumed = Agent(CodexEngine(client=client, thread_id=THREAD, max_retries=0), tools=tools, policy=policy)
    result = await resumed.run("check again")
    assert result.ok, result.error
    for capture, method in zip(launches, ("thread/start", "thread/resume"), strict=True):
        rpc = json.loads(capture.read_text())
        assert rpc["method"] == method
        params = rpc["params"]
        assert params["cwd"] == str(workspace)
        assert params["sandbox"] == "workspace-write"
        assert params["approvalPolicy"] == "never"
        assert params["config"] == compile_codex(policy)["config_overrides"]
        assert [tool["name"] for tool in params["dynamicTools"]] == ["allowed"]
    assert invoked == ["allowed", "allowed"]

    document = load_profiles()
    changed = resolve("edit", grant, replace(document, revision=document.revision + ".changed"))
    report.write_text(json.dumps(report_for(changed, revision)))
    newer = Agent(CodexEngine(client=client, thread_id=THREAD, max_retries=0), policy=changed)
    result = await newer.run("check")
    assert not result.ok
    assert "does not match the current admission" in result.error.message
    assert len(launches) == 2

    drift = True
    report.write_text(json.dumps(report_for(policy, revision)))
    result = await resumed.run("check with drift")
    assert not result.ok
    assert "Effective Codex configuration differs: MCP inventory" in result.error.message
    assert len(launches) == 3
    assert not launches[-1].exists()


@pytest.mark.asyncio
async def test_unreachable_release_canary_is_red_for_every_row(tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def unavailable():
        raise OSError("injected endpoint failure")
        yield

    monkeypatch.setattr(release_check, "codex_executable", lambda: sys.executable)
    monkeypatch.setattr(release_check, "runtime_revision", lambda _: "runtime")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "codex-cli fixture")
    monkeypatch.setattr(release_check, "_endpoint", unavailable)
    report = await release_check.run_check(load_profiles(), tmp_path)
    assert len(report["matrix"]) == 6
    assert all(row["status"] == "red" for row in report["matrix"])
    assert all("injected endpoint failure" in row["probes"]["shell_egress"]["detail"] for row in report["matrix"])
