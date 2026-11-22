from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lazybridge import Agent, Envelope, LLMEngine, Plan, Session, Step
from lazybridge.core.types import CompletionResponse, ToolCall, ToolResultContent
from lazybridge.engines.claude_code.mcp_adapter import to_mcp_tools
from lazybridge.engines.codex import CodexEngine
from lazybridge.engines.codex.app_server import CodexTurnUncertain
from lazybridge.engines.codex.dynamic_tools import dispatcher
from lazybridge.envelope import ErrorInfo
from lazybridge.guardrails import ContentGuard, GuardAction
from lazybridge.session import EventType
from lazybridge.testing import MockAgent


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ErrorInfo(type="ProviderError", message="Temporary provider failure"), "Temporary provider failure"),
        (ErrorInfo(type="GuardBlocked", message="Private moderation details"), "Blocked by the agent guard"),
    ],
)
async def test_coding_adapters_report_failed_agent_tool_results(error: ErrorInfo, message: str) -> None:
    tool = MockAgent(Envelope(payload="Unpublished draft", error=error), name="writer").as_tool()
    codex_events = []
    claude_events = []
    codex_call = dispatcher([tool], observer=lambda event, details: codex_events.append((event, details)))
    (claude_tool,) = to_mcp_tools([tool], observer=lambda event, details: claude_events.append((event, details)))

    codex_result = await codex_call("writer", {"task": "Draft a summary"})
    claude_result = await claude_tool.handler({"task": "Draft a summary"})

    text = f"Agent error ({error.type}): {message}"
    assert codex_result == {"success": False, "contentItems": [{"type": "inputText", "text": text}]}
    assert claude_result == {"isError": True, "content": [{"type": "text", "text": text}]}
    for events in (codex_events, claude_events):
        assert events[-1] == ("error", {"tool_name": "writer", "error": text})


async def test_llm_tool_results_and_events_exclude_guarded_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = Envelope(payload="Unpublished draft", error=ErrorInfo(type="GuardBlocked", message="Moderation details"))
    writer = MockAgent(blocked, name="writer")
    execute = AsyncMock(
        side_effect=[
            CompletionResponse(
                content="",
                tool_calls=[ToolCall(id="one", name="writer", arguments={"task": "Draft a summary"})],
            ),
            CompletionResponse(content="Handled the unavailable draft"),
        ],
    )
    engine = LLMEngine("fake", provider="fake")
    monkeypatch.setattr(engine, "_make_executor", lambda: SimpleNamespace(_provider=object(), aexecute=execute))

    with Session() as session:
        result = await Agent(name="parent", engine=engine, tools=[writer.as_tool()], session=session).run("Delegate")
        errors = session.events.query(event_type=EventType.TOOL_ERROR)
        assert session.events.query(event_type=EventType.TOOL_RESULT) == []

    assert result.ok
    assert execute.await_count == 2
    request = execute.await_args_list[1].args[0]
    [block] = [block for block in request.messages[-1].content if isinstance(block, ToolResultContent)]
    text = "Agent error (GuardBlocked): Blocked by the agent guard"
    assert block.content == text
    assert block.is_error is True
    assert errors[0]["payload"]["error"] == text


async def test_output_guard_returns_no_draft() -> None:
    writer = Agent(
        name="writer",
        engine=Plan(Step(MockAgent("Unpublished draft", name="draft"))),
        guard=ContentGuard(output_fn=lambda _: GuardAction.block("Draft needs approval")),
    )

    result = await writer.as_tool().run(task="Draft a summary")

    assert result.error is not None
    assert result.error.type == "GuardBlocked"
    assert result.payload is None


@pytest.mark.parametrize("as_tool", [False, True], ids=["agent", "tool"])
async def test_uncertain_codex_writer_stops_before_verify_and_fallback(as_tool: bool) -> None:
    error = CodexTurnUncertain("Inspect the writer outcome", thread_id="thread", turn_id="turn")
    call = AsyncMock(side_effect=error)
    reviewer = MockAgent("rejected", name="reviewer")
    fallback = MockAgent("fallback completed", name="fallback")
    writer = Agent(
        name="writer",
        engine=CodexEngine(client=SimpleNamespace(run=call), persist_thread=True),
        verify=None if as_tool else reviewer,
        fallback=fallback,
    )

    if as_tool:
        result = await writer.as_tool(verify=reviewer).run(task="Draft a summary")
    else:
        result = await writer.run("Draft a summary")

    assert result.error is not None
    assert result.error.type == "CodexTurnUncertain"
    assert result.error.outcome_uncertain
    assert not result.error.retryable
    assert call.await_count == 1
    assert not reviewer.calls
    assert not fallback.calls


async def test_uncertain_codex_stream_does_not_start_fallback() -> None:
    error = CodexTurnUncertain("Inspect the writer outcome", thread_id="thread", turn_id="turn")
    call = AsyncMock(side_effect=error)
    fallback = MockAgent("fallback completed", name="fallback")
    writer = Agent(
        name="writer",
        engine=CodexEngine(client=SimpleNamespace(run=call), persist_thread=True),
        fallback=fallback,
    )

    with pytest.raises(CodexTurnUncertain):
        _ = [chunk async for chunk in writer.stream("Draft a summary")]

    assert call.await_count == 1
    assert not fallback.calls


async def test_uncertain_reviewer_does_not_repeat_a_successful_writer() -> None:
    writer = MockAgent("Draft", name="writer")
    error = CodexTurnUncertain("Inspect the reviewer outcome", thread_id="thread", turn_id="turn")
    reviewer = MockAgent(Envelope.error_envelope(error), name="reviewer")
    agent = Agent(name="producer", engine=Plan(Step(writer)), verify=reviewer)

    result = await agent.run("Draft a summary")

    assert result.error is not None and result.error.outcome_uncertain
    assert len(writer.calls) == len(reviewer.calls) == 1
