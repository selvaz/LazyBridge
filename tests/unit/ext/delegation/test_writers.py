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


@pytest.mark.asyncio
async def test_claude_writer_validate_model_rejects_before_anything_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    def engine_that_must_not_be_built(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("ClaudeCodeEngine must not be constructed once validate_model refuses")

    monkeypatch.setattr("lazybridge.engines.claude_code.ClaudeCodeEngine", engine_that_must_not_be_built)
    store = Store()
    tool = make_claude_writer(
        workspace_root=Path("C:/work/project"),
        gate=_Gate(),
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Claude",
        model="gpt-5",
        validate_model=lambda model: f"REJECTED: {model} is not an Anthropic model",
    )

    result = await tool.func("implement it")

    assert result == "REJECTED: gpt-5 is not an Anthropic model"
    assert _jobs(store) == []


@pytest.mark.asyncio
async def test_codex_writer_admission_gate_refuses_after_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    engine_built = False

    def engine_that_tracks_construction(*args: Any, **kwargs: Any) -> Any:
        nonlocal engine_built
        engine_built = True
        return SimpleNamespace()

    monkeypatch.setattr("lazybridge.engines.codex.CodexEngine", engine_that_tracks_construction)

    class Channel:
        async def ask(self, prompt: str) -> bool:
            return True

    async def admission_gate() -> Any:
        return SimpleNamespace(allowed=False, reason="quota exhausted")

    store = Store()
    tool = make_codex_writer(
        workspace_root=Path("C:/work/project"),
        gate=_Gate(),
        channel=Channel(),
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Codex",
        admission_gate=admission_gate,
    )

    await tool.func("implement it")
    await _drain()

    [job] = _jobs(store)
    assert job["status"] == "failed"
    assert "quota exhausted" in job["error"]
    assert job["execution_started"] is False
    assert engine_built is False


# ---------------------------------------------------------------------------
# Gap 4 -- max_turns / writable_roots knobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_writer_max_turns_defaults_to_60_and_is_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeClaudeEngine:
        def __init__(self, *, model: str, cwd: str, config: Any, request_timeout: Any, max_turns: int) -> None:
            captured["max_turns"] = max_turns

    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> _Envelope:
            return _Envelope("claude finished")

    monkeypatch.setattr("lazybridge.engines.claude_code.ClaudeCodeEngine", FakeClaudeEngine)
    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    store = Store()
    tool = make_claude_writer(
        workspace_root=Path("C:/work/project"),
        gate=_Gate(),
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Claude",
        max_turns=100,
    )
    await tool.func("implement it")
    await _drain()
    assert captured["max_turns"] == 100


@pytest.mark.asyncio
async def test_codex_writer_writable_roots_reach_codex_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeCodexEngine:
        def __init__(self, *, model: Any, cwd: str, config: Any, request_timeout: Any) -> None:
            captured["config"] = config

    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> _Envelope:
            return _Envelope("codex finished")

    monkeypatch.setattr("lazybridge.engines.codex.CodexEngine", FakeCodexEngine)
    monkeypatch.setattr("lazybridge.Agent", FakeAgent)

    class Channel:
        async def ask(self, prompt: str) -> bool:
            return True

    store = Store()
    tool = make_codex_writer(
        workspace_root=Path("C:/work/project"),
        gate=_Gate(),
        channel=Channel(),
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Codex",
        writable_roots=["C:/work/.git/worktrees/project"],
    )
    await tool.func("implement it")
    await _drain()

    assert captured["config"].codex.writable_roots == ("C:/work/.git/worktrees/project",)


# ---------------------------------------------------------------------------
# Gap 2 -- per-call model/effort/session overrides on the writers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_writer_accepts_per_call_model_effort_and_session_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeClaudeEngine:
        def __init__(
            self,
            *,
            model: str,
            cwd: str,
            config: Any,
            request_timeout: Any,
            max_turns: int,
            reasoning_effort: str | None = None,
            session_id: str | None = None,
        ) -> None:
            captured.update(model=model, reasoning_effort=reasoning_effort, session_id=session_id)

    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> _Envelope:
            return _Envelope("claude finished")

    monkeypatch.setattr("lazybridge.engines.claude_code.ClaudeCodeEngine", FakeClaudeEngine)
    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    store = Store()
    tool = make_claude_writer(
        workspace_root=Path("C:/work/project"),
        gate=_Gate(),
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Claude",
        accept_model_override=True,
        accept_effort_override=True,
        accept_session_override=True,
    )
    await tool.func(objective="implement it", model="opus", effort="high", session="resume-me")
    await _drain()

    assert captured == {"model": "opus", "reasoning_effort": "high", "session_id": "resume-me"}


@pytest.mark.asyncio
async def test_claude_writer_without_overrides_builds_engine_with_the_pre_1_8_kwargs_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No accept_*_override means ClaudeCodeEngine is built with exactly the
    same keyword arguments as before those existed -- proven here against
    a stand-in engine with the OLD, narrower constructor signature."""
    captured: dict[str, Any] = {}

    class FakeClaudeEngine:
        def __init__(self, *, model: str, cwd: str, config: Any, request_timeout: Any, max_turns: int) -> None:
            captured.update(model=model, cwd=cwd, max_turns=max_turns)

    class FakeAgent:
        def __init__(self, *, engine: Any, name: str) -> None:
            pass

        async def run(self, objective: str) -> _Envelope:
            return _Envelope("claude finished")

    monkeypatch.setattr("lazybridge.engines.claude_code.ClaudeCodeEngine", FakeClaudeEngine)
    monkeypatch.setattr("lazybridge.Agent", FakeAgent)
    store = Store()
    tool = make_claude_writer(
        workspace_root=Path("C:/work/project"),
        gate=_Gate(),
        registry=JobRegistry(store),
        background_tasks=set(),
        doc="write with Claude",
    )
    await tool.func("implement it")
    await _drain()
    assert captured["model"] == "sonnet"
