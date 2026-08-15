from __future__ import annotations

import asyncio

from lazybridge import Session
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
