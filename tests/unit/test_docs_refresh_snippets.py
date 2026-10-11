"""Execute the actual changed Markdown examples with fake model transports.

Real Agent/Plan/config/registry/tool paths are retained. No SDK subprocess or
provider call is allowed. Fragment setup mirrors the surrounding guide text.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import struct
import sys
import tempfile
import tomllib
import types
import zlib
from pathlib import Path
from typing import Any

import pytest

import lazybridge
from lazybridge import Envelope, Session, Store
from lazybridge.engines.claude_code.protocol import ClaudeSdkResult
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient
from lazybridge.engines.codex import usage as codex_usage
from lazybridge.engines.codex.app_server import CodexAppServerClient, CodexRunResult
from lazybridge.engines.llm import LLMEngine
from lazybridge.testing import MockAgent

ROOT = Path(__file__).resolve().parents[2]
FENCE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.M | re.S)
MANIFEST = json.loads((Path(__file__).with_name("docs_refresh_snippets.json")).read_text(encoding="utf-8"))


def sample_schema(schema, root=None):
    root = root or schema
    if "$ref" in schema:
        return sample_schema(root["$defs"][schema["$ref"].split("/")[-1]], root)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema:
        return sample_schema(schema["anyOf"][0], root)
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {key: sample_schema(value, root) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return [sample_schema(schema.get("items", {"type": "string"}), root)]
    if kind == "boolean":
        return True
    if kind in {"integer", "number"}:
        return 100
    if kind == "null":
        return None
    return "review complete"


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    calls = {"claude": [], "codex": [], "llm": [], "claude_attachments": []}

    async def claude_run(self, prompt, *, options, attachments=()):
        calls["claude"].append((prompt, options))
        calls["claude_attachments"].append(attachments)
        native_id = options.resume or "fake-claude-session"
        if options.on_session_id:
            options.on_session_id(native_id)
        text = (
            json.dumps(sample_schema(options.output_format["schema"])) if options.output_format else "Corrected draft"
        )
        for tool in options.mcp_tools:
            if tool.name == "project_status":
                result = await tool.handler({})
                assert result["content"]
        return ClaudeSdkResult(text=text, session_id=native_id, input_tokens=10, output_tokens=5)

    async def codex_run(self, **kw):
        calls["codex"].append(kw)
        native_id = kw.get("thread_id") or "fake-codex-thread"
        if kw.get("progress") is not None:
            kw["progress"].update(thread_id=native_id, turn_sent=True)
        system = kw.get("developer_instructions") or ""
        text = "approved" if "approved" in system else "Corrected draft"
        if kw.get("output_schema"):
            text = json.dumps(sample_schema(kw["output_schema"]))
        for tool in kw.get("dynamic_tools", []):
            if tool["name"] == "project_status":
                result = await kw["on_tool_call"]("project_status", {})
                assert result["success"]
        if kw.get("on_text"):
            await kw["on_text"](text)
        return CodexRunResult(text=text, thread_id=native_id, input_tokens=10, output_tokens=5)

    async def llm_run(self, env, *, output_type=str, **kw):
        calls["llm"].append(env)
        payload = "approved"
        if hasattr(output_type, "model_json_schema"):
            payload = output_type.model_validate(sample_schema(output_type.model_json_schema()))
        return Envelope(task=env.task, payload=payload)

    async def llm_stream(self, env, **kw):
        yield (await llm_run(self, env, **kw)).text()

    def forbidden(*args, **kwargs):
        raise AssertionError("Documentation attempted an unmocked runtime/network call")

    async def fetch_usage():
        return codex_usage.parse_rate_limits(
            {
                "rateLimits": {
                    "limitId": "codex",
                    "secondary": {
                        "usedPercent": 25,
                        "windowDurationMins": 10080,
                    },
                }
            }
        )

    monkeypatch.setattr(AgentSdkClient, "run", claude_run)
    monkeypatch.setattr(CodexAppServerClient, "run", codex_run)
    monkeypatch.setattr(codex_usage, "fetch_codex_usage", fetch_usage)
    monkeypatch.setattr(LLMEngine, "run", llm_run)
    monkeypatch.setattr(LLMEngine, "stream", llm_stream)
    monkeypatch.setattr(lazybridge.ClaudeCodeEngine, "_tag_new_session", lambda *args: None)
    monkeypatch.setattr(AgentSdkClient, "stream", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr("builtins.input", lambda *args: "continue")
    # Native-history listing is read-only, but must not inspect the operator's
    # real transcripts when executing the guide's retention example.
    sdk = types.ModuleType("claude_agent_sdk")
    sdk.list_sessions = lambda: [types.SimpleNamespace(session_id="fake-claude-session", tag="lazybridge")]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", sdk)
    return calls


def fragment_setup():
    ns = {name: getattr(lazybridge, name) for name in lazybridge.__all__ if name != "CacheConfig"}
    ns.update(
        Any=Any,
        asyncio=asyncio,
        session=Session(),
        sess=Session(),
        store=Store(),
        task="Review the draft",
        repo=str(Path.cwd()),
        my_validator=lambda value: value,
    )
    for name in ("a", "b", "c", "search", "researcher", "writer", "researcher_a", "researcher_b", "researcher_c"):
        native_name = {"researcher": "research", "writer": "write"}.get(name, name)
        ns[name] = MockAgent("approved", name=native_name)
    return ns


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda e: e["path"])
def test_changed_markdown_examples_execute(entry, offline):
    codes = FENCE.findall((ROOT / entry["path"]).read_text(encoding="utf-8"))
    indexes = list(range(len(codes))) if entry.get("all") else entry["blocks"]
    ns = fragment_setup()
    for index in indexes:
        code = codes[index]
        try:
            exec(compile(code, f"{entry['path']}:block-{index}", "exec", dont_inherit=True), ns)
        except Exception as exc:
            pytest.fail(f"{entry['path']} block {index}: {type(exc).__name__}: {exc}")
        # A swallowed model/Plan failure must not pass as successful execution.
        for key in ("result", "env"):
            value = ns.get(key)
            if isinstance(value, Envelope):
                assert value.ok, f"{entry['path']} block {index}: {value.error}"


def test_verification_feedback_exhaustion_and_shared_state(offline):
    codes = FENCE.findall((ROOT / "docs/guides/full/claude-writes-codex-reviews.md").read_text(encoding="utf-8"))
    ns = {}
    exec(codes[1], ns)
    assert ns["reviewer"].call_count == 2
    producer = ns["writer"].engine.mock
    assert producer.call_count == 2
    assert producer.calls[0].task == producer.calls[1].task
    assert "Sunday" in producer.calls[1].context
    exec(codes[2], ns)
    assert ns["result"].ok  # Execution success is not an approval flag.

    ns = {}
    exec(FENCE.findall((ROOT / "docs/index.md").read_text(encoding="utf-8"))[0], ns)
    assert set(ns["claude"]._tool_map) == set(ns["codex"]._tool_map) == {"project_status"}


def test_refreshed_install_commands_use_real_extras_and_current_version():
    """Validate install snippets without installing into the shared interpreter."""
    from packaging.requirements import Requirement

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for entry in MANIFEST:
        if not entry.get("all"):
            continue
        page = (ROOT / entry["path"]).read_text(encoding="utf-8")
        for block in re.findall(r"^```(?:sh|bash)[^\n]*\n(.*?)^```", page, re.M | re.S):
            for line in block.splitlines():
                command = shlex.split(line)
                if command[:4] == ["python", "-m", "pip", "install"]:
                    for value in command[4:]:
                        requirement = Requirement(value)
                        assert requirement.name == "lazybridge"
                        assert requirement.extras <= project["optional-dependencies"].keys()
                        assert project["version"] in requirement.specifier
                elif command[:3] == ["python", "-m", "pytest"]:
                    assert (ROOT / command[3]).exists()
                elif command[:1] == ["python"]:
                    # Authenticated live smoke scripts are opt-in, never run
                    # by the documentation tests; validate their checked-in path.
                    assert len(command) == 2
                    assert command[1].startswith(("examples/claude_code/", "examples/codex/"))
                    assert (ROOT / command[1]).is_file()
                else:
                    assert command == ["claude", "doctor"]


@pytest.mark.parametrize("path", [
    "docs/guides/full/claude-code-engine.md",
    "docs/guides/full/codex-engine.md",
])
def test_restored_image_examples_reach_the_transport(path, offline):
    """A successful text answer alone must not hide a dropped attachment."""
    codes = FENCE.findall((ROOT / path).read_text(encoding="utf-8"))
    code = next(code for code in codes if "ImageContent.from_bytes" in code)
    ns = {}
    exec(compile(code, path, "exec", dont_inherit=True), ns)
    # Validate the fixture without requiring Pillow. Mock text transports alone
    # would accept a damaged PNG that a real vision call might reject.
    raw = ns["png"]
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    while position < len(raw):
        length = struct.unpack(">I", raw[position:position + 4])[0]
        body = raw[position + 4:position + 8 + length]
        crc = struct.unpack(">I", raw[position + 8 + length:position + 12 + length])[0]
        assert zlib.crc32(body) == crc
        position += length + 12
    if "claude-code" in path:
        attachments = offline["claude_attachments"][-1]
    else:
        attachments = offline["codex"][-1]["attachments"]
    assert attachments, "Image example did not reach its native adapter"


def test_restoration_manifest_covers_every_touched_python_fence():
    """The restored narrative pages must execute all examples, not selected fragments."""
    restored = {
        "docs/best-python-ai-agent-frameworks.md", "docs/comparison.md", "docs/why.md",
        "docs/concepts/mental-model.md", "docs/for-llms/codegen-contract.md",
        "docs/guides/full/claude-code-engine.md", "docs/guides/full/codex-engine.md",
        "docs/guides/full/coding-agent-config.md", "docs/guides/full/coding-agents.md",
        "docs/index.md", "docs/quickstart.md", "README.md",
    }
    assert restored <= {entry["path"] for entry in MANIFEST if entry.get("all")}
    checkpoint = next(entry for entry in MANIFEST if entry["path"].endswith("/checkpoint.md"))
    # Block 0 is a deliberately schematic constructor signature, left untouched.
    assert checkpoint["blocks"] == list(range(1, len(FENCE.findall(
        (ROOT / checkpoint["path"]).read_text(encoding="utf-8")
    ))))


def test_llmstxt_config_excerpt_matches_the_build():
    """The restored YAML recipe selects actual configured sections."""
    import yaml

    page = (ROOT / "docs/for-llms/llms-txt.md").read_text(encoding="utf-8")
    code = re.search(r"^```yaml\n(.*?)^```", page, re.M | re.S).group(1)
    excerpt = yaml.safe_load(code)["plugins"][0]["llmstxt"]
    # BaseLoader allows the project's Python-name theme tags without executing them.
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    configured = next(p["llmstxt"] for p in config["plugins"] if isinstance(p, dict) and "llmstxt" in p)
    assert excerpt["full_output"] == configured["full_output"]
    assert excerpt["sections"].items() <= configured["sections"].items()


def test_refreshed_machine_readable_patterns_execute(offline, tmp_path):
    from pydantic import BaseModel

    class Status(BaseModel):
        ready: bool

    contract = json.loads((ROOT / "lazybridge/llms.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert contract["version"] == project["version"]
    ns = fragment_setup()
    ns.update(
        get_weather=lambda city: "sunny",
        analyst=MockAgent("analysis", name="analyst"),
        MyPydanticModel=Status,
        registry=lazybridge.SessionRegistry(path=tmp_path / "sessions.json"),
        repo=str(tmp_path),
    )
    exec(contract["canonical_import"], ns)
    for key in (
        "single_agent",
        "agent_with_tool",
        "parallel_fanout",
        "coding_review",
        "named_claude",
        "named_codex",
        "structured_output",
    ):
        exec(contract["canonical_patterns"][key], ns)
    assert isinstance(ns["env"].payload, str)
