from __future__ import annotations

from collections.abc import AsyncIterator

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
from lazybridge.ext.approval import TieredGate
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
