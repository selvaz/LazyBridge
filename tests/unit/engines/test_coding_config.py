from __future__ import annotations

import asyncio

import pytest

from lazybridge import Session
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.codex import CodexEngine
from lazybridge.engines.coding import (
    ApprovalDecision,
    ApprovalRequest,
    CodingAgentConfig,
    ask_approval,
    remembering_gate,
    session_approvals,
)


def test_reviewer_is_read_only_and_fails_closed_for_application_tools():
    config = CodingAgentConfig.reviewer()

    assert config.claude.preapprove_application_tools is False
    assert config.codex.sandbox == "read-only"
    assert config.codex.approval_policy == "never"
    assert config.codex.preapprove_dynamic_tools is False
    assert config.approval_gate is None


def test_writer_uses_native_on_request_profiles_and_shared_gate():
    async def gate(request):
        return ApprovalDecision.allow()

    config = CodingAgentConfig.writer(gate)

    assert config.claude.permission_mode == "default"
    assert config.claude.preapprove_application_tools is False
    assert config.codex.sandbox == "workspace-write"
    assert config.codex.approval_policy == "on-request"
    assert config.codex.preapprove_dynamic_tools is False
    assert config.approval_gate is gate


def test_ask_approval_accepts_sync_gate_and_fails_closed_without_one():
    request = ApprovalRequest(provider="codex", kind="command", name="git status")

    allowed = asyncio.run(ask_approval(lambda _: ApprovalDecision.allow(), request))
    denied = asyncio.run(ask_approval(None, request))

    assert allowed.action == "allow"
    assert denied.action == "deny"


def test_allow_session_is_remembered_per_agent_and_session_not_per_run():
    """``allow_session`` must survive across runs of the same agent.

    Scoping it to one run (the natural place to keep a cache, since the
    dispatcher is rebuilt per run) would re-prompt the user on every turn
    while still calling itself "session".
    """
    asked: list[str] = []

    async def gate(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(f"{request.provider}:{request.name}")
        return ApprovalDecision.allow_for_session()

    session = Session()
    request = ApprovalRequest(provider="codex", kind="tool", name="get_quote")

    async def two_runs() -> None:
        for _ in range(2):
            scoped = remembering_gate(gate, session_approvals(session, "codex", "analyst"))
            assert (await ask_approval(scoped, request)).action == "allow_session"

    asyncio.run(two_runs())

    assert asked == ["codex:get_quote"]  # asked once, not once per run


@pytest.mark.parametrize(
    ("provider", "name"),
    [("codex", "codex-shell"), ("claude-code", "Bash")],
)
def test_allow_session_for_command_does_not_approve_different_command(provider, name):
    asked: list[str] = []

    async def gate(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.arguments["command"])
        return ApprovalDecision.allow_for_session()

    scoped = remembering_gate(gate, set())
    first = ApprovalRequest(provider=provider, kind="command", name=name, arguments={"command": "git status"})
    second = ApprovalRequest(provider=provider, kind="command", name=name, arguments={"command": "rm -rf important-dir"})

    async def approve_both() -> None:
        assert (await scoped(first)).action == "allow_session"
        assert (await scoped(second)).action == "allow_session"

    asyncio.run(approve_both())

    assert asked == ["git status", "rm -rf important-dir"]


@pytest.mark.parametrize(
    ("provider", "name"),
    [("codex", "codex-shell"), ("claude-code", "Bash")],
)
def test_allow_session_for_command_is_reused_for_identical_command(provider, name):
    asked: list[str] = []

    async def gate(request: ApprovalRequest) -> ApprovalDecision:
        asked.append(request.arguments["command"])
        return ApprovalDecision.allow_for_session()

    scoped = remembering_gate(gate, set())
    request = ApprovalRequest(provider=provider, kind="command", name=name, arguments={"command": "git status"})

    async def approve_twice() -> None:
        assert (await scoped(request)).action == "allow_session"
        assert (await scoped(request)).action == "allow_session"

    asyncio.run(approve_twice())

    assert asked == ["git status"]


def test_each_agent_and_provider_keeps_its_own_approvals():
    """A shared engine instance must not leak one agent's grant to another."""

    async def gate(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.allow_for_session()

    session = Session()
    request = ApprovalRequest(provider="codex", kind="tool", name="get_quote")

    async def grant_for(agent: str, provider: str) -> None:
        scoped = remembering_gate(gate, session_approvals(session, provider, agent))
        await ask_approval(scoped, request)

    asyncio.run(grant_for("analyst", "codex"))

    assert session_approvals(session, "codex", "analyst") == {("tool", "get_quote")}
    assert session_approvals(session, "codex", "auditor") == set()
    assert session_approvals(session, "claude-code", "analyst") == set()


def test_without_a_session_the_grant_degrades_to_the_current_run():
    """No Session means nowhere to persist — it must not crash or leak globally."""
    first = session_approvals(None, "codex", "analyst")
    first.add(("tool", "get_quote"))

    assert session_approvals(None, "codex", "analyst") == set()


@pytest.mark.parametrize("engine_type", [ClaudeCodeEngine, CodexEngine])
def test_engines_accept_approval_gate_as_a_direct_keyword(engine_type):
    async def gate(request):
        return ApprovalDecision.allow()

    engine = engine_type(approval_gate=gate)

    assert engine.approval_gate is gate


@pytest.mark.parametrize("engine_type", [ClaudeCodeEngine, CodexEngine])
def test_engines_preserve_a_falsy_direct_approval_gate(engine_type):
    class FalsyGate:
        def __len__(self):
            return 0

        async def __call__(self, request):
            return ApprovalDecision.deny()

    gate = FalsyGate()
    engine = engine_type(approval_gate=gate)

    assert engine.approval_gate is gate


@pytest.mark.parametrize("engine_type", [ClaudeCodeEngine, CodexEngine])
def test_engines_reject_conflicting_direct_and_configured_gates(engine_type):
    async def direct_gate(request):
        return ApprovalDecision.allow()

    async def configured_gate(request):
        return ApprovalDecision.allow()

    with pytest.raises(ValueError, match="approval_gate conflicts"):
        engine_type(
            approval_gate=direct_gate,
            config=CodingAgentConfig(approval_gate=configured_gate),
        )
