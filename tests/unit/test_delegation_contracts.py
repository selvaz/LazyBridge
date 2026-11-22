from __future__ import annotations

import asyncio
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
from lazybridge.exporters import CallbackExporter
from lazybridge.guardrails import ContentGuard, GuardAction
from lazybridge.session import EventType
from lazybridge.signals import conclude
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


async def test_guarded_plan_step_keeps_draft_out_of_stream_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    draft = "Unpublished draft"
    awaiting_review = asyncio.Event()
    review = asyncio.Event()
    exported = []
    redactor_inputs = []

    class Review(ContentGuard):
        async def acheck_output(self, text: str) -> GuardAction:
            assert text == draft
            awaiting_review.set()
            await review.wait()
            return GuardAction.block(f"Review rejected {draft}")

    async def fake_loop(env, *, session, run_id, _stream_sink, **kwargs):
        session.emit(EventType.MODEL_RESPONSE, {"content": draft, "input_tokens": 1, "output_tokens": 2}, run_id=run_id)
        if _stream_sink is not None:
            await _stream_sink.put(draft)
        return Envelope(task=env.task, payload=draft)

    def redact(payload):
        redactor_inputs.append(payload)
        return payload

    engine = LLMEngine("fake", provider="fake")
    monkeypatch.setattr(engine, "_loop", fake_loop)
    chunks = []
    with Session(redact=redact, exporters=[CallbackExporter(fn=exported.append)]) as session:
        writer = Agent(name="writer", engine=engine, guard=Review(), session=session)
        parent = Agent(name="parent", engine=Plan(Step(writer)), session=session)

        async def consume():
            async for chunk in parent.stream("Draft a summary"):
                chunks.append(chunk)

        pending = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(awaiting_review.wait(), 2.0)
            session.emit(EventType.MODEL_RESPONSE, {"content": "Unrelated response"})
            assert not chunks
            assert "Unrelated response" in str(exported)
            assert draft not in str(exported)
        finally:
            review.set()
            await asyncio.wait_for(pending, 2.0)
        rows = session.events.query()
        [decision] = session.events.query(event_type=EventType.GUARD_DECISION)
        [parent_error] = session.events.query(event_type=EventType.TOOL_ERROR)

    assert draft not in str((chunks, exported, rows, redactor_inputs))
    assert any(row["payload"].get("output_tokens") == 2 for row in rows)
    assert decision["payload"]["allowed"] is False
    assert parent_error["payload"]["error"] == "Agent error (GuardBlocked): Blocked by the agent guard"


async def test_guarded_stream_applies_parent_guard_to_fallback_before_yield() -> None:
    class BrokenStream:
        async def stream(self, *args, **kwargs):
            raise RuntimeError("provider down")
            yield

    class GoodStream:
        async def stream(self, *args, **kwargs):
            yield "private fallback draft"

    fallback = Agent(name="fallback", engine=GoodStream())
    primary = Agent(
        name="primary",
        engine=BrokenStream(),
        fallback=fallback,
        guard=ContentGuard(output_fn=lambda _: GuardAction.block("Review denied")),
    )

    with pytest.raises(ValueError, match="Review denied"):
        async for chunk in primary.stream("Draft a summary"):
            pytest.fail(f"Guard should have withheld {chunk!r}")


async def test_guarded_conclude_cannot_publish_denied_message() -> None:
    async def finish(task: str) -> str:
        return conclude("Unpublished draft")

    guarded = Agent(
        name="writer",
        engine=Plan(Step(finish)),
        guard=ContentGuard(output_fn=lambda _: GuardAction.block("Review denied")),
    )
    result = await guarded.run("Draft a summary")
    assert result.error is not None and result.error.type == "GuardBlocked"
    assert result.payload is None


async def test_output_guard_publishes_the_rewritten_text() -> None:
    writer = Agent(
        name="writer",
        engine=Plan(Step(MockAgent("Unpublished draft", name="draft"))),
        guard=ContentGuard(output_fn=lambda _: GuardAction.modify("Approved summary")),
    )

    assert (await writer.run("Draft a summary")).text() == "Approved summary"
    assert [chunk async for chunk in writer.stream("Draft a summary")] == ["Approved summary"]


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
