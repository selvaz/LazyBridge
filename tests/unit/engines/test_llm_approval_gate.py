from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from lazybridge.core.types import (
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
    ToolCall,
    ToolResultContent,
    UsageStats,
)
from lazybridge.engines.coding import ApprovalDecision, ApprovalRequest
from lazybridge.engines.llm import LLMEngine
from lazybridge.envelope import Envelope
from lazybridge.ext.approval import Rule, TieredGate
from lazybridge.session import EventType, Session
from lazybridge.tools import Tool


async def _run_tool(engine: LLMEngine, tool: Tool, session: Session | None = None):
    return await engine._exec_tool(
        ToolCall(id="call-1", name=tool.name, arguments={}),
        {tool.name: tool},
        agent_name="test-agent",
        session=session,
        run_id="run-1",
    )


@pytest.mark.asyncio
async def test_no_gate_preserves_ungated_tool_execution():
    calls = 0

    def ping() -> str:
        nonlocal calls
        calls += 1
        return "pong"

    result = await _run_tool(LLMEngine("fake", provider="fake"), Tool(ping))

    assert result == "pong"
    assert calls == 1


@pytest.mark.asyncio
async def test_allow_gate_executes_tool_once_with_normalized_request():
    calls = 0
    requests: list[ApprovalRequest] = []

    def ping() -> str:
        nonlocal calls
        calls += 1
        return "pong"

    async def gate(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision.allow()

    result = await _run_tool(LLMEngine("fake", provider="fake", approval_gate=gate), Tool(ping))

    assert result == "pong"
    assert calls == 1
    assert requests == [ApprovalRequest(provider="llm", kind="tool", name="ping", arguments={})]
    assert requests[0].cwd is None


@pytest.mark.asyncio
async def test_cwd_is_resolved_and_forwarded_to_approval_request(tmp_path: Path):
    requests: list[ApprovalRequest] = []
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def gate(request: ApprovalRequest) -> ApprovalDecision:
        requests.append(request)
        return ApprovalDecision.allow()

    engine = LLMEngine("fake", provider="fake", approval_gate=gate, cwd=workspace)
    result = await _run_tool(engine, Tool(lambda: "pong", name="ping"))

    assert result == "pong"
    assert engine.cwd == str(workspace.resolve())
    assert requests[0].cwd == str(workspace.resolve())


@pytest.mark.asyncio
async def test_tiered_session_grant_does_not_cross_llm_engine_cwd(tmp_path: Path):
    prompts: list[str] = []

    class ApprovingChannel:
        name = "test"

        async def ask(self, prompt: str) -> bool:
            prompts.append(prompt)
            return True

    gate = TieredGate(channel=ApprovingChannel(), rules=(Rule("session", "ping"),))
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = LLMEngine("fake", provider="fake", approval_gate=gate, cwd=first_workspace)
    second = LLMEngine("fake", provider="fake", approval_gate=gate, cwd=second_workspace)
    tool = Tool(lambda: "pong", name="ping")

    assert await _run_tool(first, tool) == "pong"
    assert await _run_tool(second, tool) == "pong"
    assert await _run_tool(first, tool) == "pong"

    assert len(prompts) == 2
    assert [record.action for record in gate.log] == ["allow_session", "allow_session", "allow"]
    assert [record.cwd for record in gate.log] == [first.cwd, second.cwd, first.cwd]


@pytest.mark.asyncio
async def test_default_deny_never_runs_tool_and_emits_tool_error():
    calls = 0

    def ping() -> str:
        nonlocal calls
        calls += 1
        return "pong"

    gate = TieredGate(channel=object(), rules=())
    session = Session()
    result = await _run_tool(LLMEngine("fake", provider="fake", approval_gate=gate), Tool(ping), session)

    assert isinstance(result, PermissionError)
    assert "no rule" in str(result)
    assert calls == 0
    session.flush()
    errors = session.events.query(event_type=EventType.TOOL_ERROR)
    assert len(errors) == 1
    assert errors[0]["payload"]["type"] == "PermissionError"


@pytest.mark.asyncio
async def test_denial_returns_an_error_tool_result_to_the_model():
    requests: list[CompletionRequest] = []
    calls = 0

    def ping() -> str:
        nonlocal calls
        calls += 1
        return "pong"

    class FakeExecutor:
        _provider = object()

        async def aexecute(self, request: CompletionRequest) -> CompletionResponse:
            requests.append(request)
            if len(requests) == 1:
                return CompletionResponse(
                    content="",
                    tool_calls=[ToolCall(id="call-1", name="ping", arguments={})],
                    stop_reason="tool_use",
                    usage=UsageStats(),
                    model="fake",
                )
            return CompletionResponse(content="denial handled", usage=UsageStats(), model="fake")

        async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
            response = await self.aexecute(request)
            yield StreamChunk(
                delta=response.content,
                tool_calls=response.tool_calls,
                stop_reason=response.stop_reason,
                usage=response.usage,
                is_final=True,
            )

    engine = LLMEngine(
        "fake",
        provider="fake",
        approval_gate=lambda _: ApprovalDecision.deny("blocked by policy"),
    )
    engine._make_executor = lambda: FakeExecutor()  # type: ignore[assignment]

    result = await engine.run(
        Envelope(task="call ping"), tools=[Tool(ping)], output_type=str, memory=None, session=None
    )

    assert result.text() == "denial handled"
    assert calls == 0
    blocks = [
        block
        for message in requests[1].messages
        if not isinstance(message.content, str)
        for block in message.content
        if isinstance(block, ToolResultContent)
    ]
    assert len(blocks) == 1
    assert blocks[0].is_error is True
    assert blocks[0].content == "Tool error: blocked by policy"
