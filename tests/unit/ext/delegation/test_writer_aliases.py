from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lazybridge import Store
from lazybridge.engines.sessions import SessionRegistry
from lazybridge.ext.delegation import (
    ExtraParam,
    JobRegistry,
    make_claude_writer,
    make_claude_writer_engine_factory,
    make_codex_writer,
    make_codex_writer_engine_factory,
)


@pytest.mark.parametrize("kind", ["codex", "claude"])
@pytest.mark.parametrize("per_call", [False, True])
async def test_writer_resumes_alias_after_registry_and_writer_restart(tmp_path, monkeypatch, kind, per_call):
    # Use real engine constructors and their AliasBinding resolution, but no
    # provider process: fake Agent simulates the provider returning a native id.
    path = tmp_path / "sessions.json"
    seen = []

    class Agent:
        def __init__(self, *, engine, name):
            self.engine = engine

        async def run(self, objective):
            engine = self.engine
            native_id = engine.thread_id if kind == "codex" else engine.session_id
            seen.append((engine.session_alias, native_id))
            # Simulate the durable binding a completed provider turn publishes.
            engine._alias.bind("provider-id-123", model=engine.model, effort=engine.reasoning_effort)
            return SimpleNamespace(ok=True, text=lambda: "done")

    monkeypatch.setattr("lazybridge.Agent", Agent)

    class Channel:
        async def ask(self, prompt):
            return True

    for _restart in range(2):
        tasks = set()
        kwargs = dict(
            workspace_root=tmp_path,
            gate=object(),
            registry=JobRegistry(Store(db=str(tmp_path / "jobs.db"))),
            background_tasks=tasks,
            doc="write",
            session_registry=SessionRegistry(path),
            session_alias=None if per_call else "review",
            accept_session_alias_override=per_call,
        )
        if kind == "codex":
            tool = make_codex_writer(channel=Channel(), **kwargs)
        else:
            tool = make_claude_writer(**kwargs)
        call = {"session_alias": "review"} if per_call else {}
        await tool.func("work", **call)
        await asyncio.gather(*tasks)
    assert seen == [("review", None), ("review", "provider-id-123")]
    assert SessionRegistry(path).resolve(kind, tmp_path, "review") == "provider-id-123"


@pytest.mark.parametrize("kind", ["codex", "claude"])
def test_reusable_factories_accept_per_call_settings_and_scope(tmp_path, monkeypatch, kind):
    captured = []

    def fake_engine(**kwargs):
        captured.append(kwargs)
        return object()

    module = "codex.CodexEngine" if kind == "codex" else "claude_code.ClaudeCodeEngine"
    monkeypatch.setattr("lazybridge.engines." + module, fake_engine)
    registry = SessionRegistry(tmp_path / "sessions.json")
    builder = make_codex_writer_engine_factory if kind == "codex" else make_claude_writer_engine_factory
    factory = builder(workspace_root=tmp_path, gate=object(), session_registry=registry)
    settings = {"writable_roots": [str(tmp_path / "git")]} if kind == "codex" else {"max_turns": 100}
    first = factory(cwd=tmp_path / "other", model="custom", effort="high", session_alias="review", **settings)
    second = factory()
    assert first is not second
    assert captured[0]["cwd"] == str(tmp_path / "other")
    assert captured[0]["model"] == "custom"
    assert captured[0]["reasoning_effort"] == "high"
    assert captured[0]["session_alias"] == "review"
    assert captured[0]["session_registry"] is registry
    assert "session_alias" not in captured[1]
    assert captured[1]["cwd"] == str(tmp_path)
    if kind == "codex":
        assert captured[0]["config"].codex.writable_roots == (str(tmp_path / "git"),)
        assert not captured[1]["config"].codex.writable_roots
    else:
        assert [c["max_turns"] for c in captured] == [100, 60]
    with pytest.raises(ValueError, match="native session id or a session_alias"):
        factory(session="native-id", session_alias="alias")


async def test_writer_forwards_factory_extra_params_and_guard(tmp_path):
    calls = []
    registry = JobRegistry(Store(db=str(tmp_path / "jobs.db")))

    def guard(objective, kwargs):
        calls.append((objective, kwargs))
        return "REJECTED: caller policy"

    tool = make_claude_writer(
        workspace_root=tmp_path,
        gate=object(),
        registry=registry,
        background_tasks=set(),
        doc="write",
        engine_factory=lambda **kw: pytest.fail("blocked"),
        extra_params={"cwd": ExtraParam(required=True)},
        guard=guard,
    )
    assert await tool.func("work", cwd="chosen") == "REJECTED: caller policy"
    assert calls == [("work", {"cwd": "chosen"})]
    assert list(registry._store.items(prefix=registry._prefix)) == []
