from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lazybridge import Store
from lazybridge.ext.delegation.jobs import JobRegistry
from lazybridge.ext.delegation.writers import make_claude_writer, make_codex_writer


async def _drain() -> None:
    await asyncio.sleep(0.05)


class _Envelope:
    ok = True

    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text


class _Gate:
    pass


def _jobs(store: Store) -> list[dict[str, Any]]:
    return [raw for _key, raw in store.items(prefix="delegation:job:") if isinstance(raw, dict)]


@pytest.mark.asyncio
async def test_codex_writer_is_confirmed_fire_and_forget(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeCodexEngine:
        def __init__(self, *, model: Any, cwd: str, config: Any, request_timeout: Any) -> None:
            captured.update(model=model, cwd=cwd, config=config, request_timeout=request_timeout)

    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            captured["engine"] = engine

        async def run(self, objective: str) -> _Envelope:
            captured["objective"] = objective
            return _Envelope("codex finished")

    class Channel:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def ask(self, prompt: str) -> bool:
            self.prompts.append(prompt)
            return True

    monkeypatch.setattr("lazybridge.engines.codex.CodexEngine", FakeCodexEngine)
    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    gate = _Gate()
    channel = Channel()
    store = Store()
    workspace_root = Path("C:/work/project")
    tool = make_codex_writer(
        workspace_root=workspace_root,
        gate=gate,
        channel=channel,
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Codex",
    )

    immediate = await tool.func("implement it")
    assert "Requested approval" in immediate
    assert "codex finished" not in immediate
    assert _jobs(store)[0]["status"] == "awaiting_approval"
    await _drain()

    [job] = _jobs(store)
    assert job["status"] == "done" and job["result"] == "codex finished"
    assert channel.prompts == ["About to delegate to Codex: implement it\n\nProceed?"]
    assert captured["config"].approval_gate is gate
    assert captured["config"].codex.sandbox == "workspace-write"
    assert captured["config"].codex.approval_policy == "on-request"
    assert captured["config"].codex.preapprove_dynamic_tools is False
    assert captured["request_timeout"] is None


@pytest.mark.asyncio
async def test_claude_writer_starts_running_without_preconfirm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClaudeEngine:
        def __init__(self, *, model: str, cwd: str, config: Any, request_timeout: Any, max_turns: int) -> None:
            captured.update(
                model=model,
                cwd=cwd,
                config=config,
                request_timeout=request_timeout,
                max_turns=max_turns,
            )

    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            captured["engine"] = engine

        async def run(self, objective: str) -> _Envelope:
            captured["objective"] = objective
            return _Envelope("claude finished")

    monkeypatch.setattr("lazybridge.engines.claude_code.ClaudeCodeEngine", FakeClaudeEngine)
    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    gate = _Gate()
    store = Store()
    workspace_root = Path("C:/work/project")
    tool = make_claude_writer(
        workspace_root=workspace_root,
        gate=gate,
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Claude",
    )

    immediate = await tool.func("implement it")
    assert "Started job" in immediate
    assert "claude finished" not in immediate
    assert _jobs(store)[0]["status"] == "running"
    await _drain()

    [job] = _jobs(store)
    assert job["status"] == "done" and job["result"] == "claude finished"
    assert captured["config"].approval_gate is gate
    assert captured["config"].claude.permission_mode == "default"
    assert captured["config"].claude.preapprove_application_tools is False
    assert captured["config"].claude.extra_tools == ("Write", "Edit", "Bash")
    assert captured["request_timeout"] is None
    assert captured["max_turns"] == 60
