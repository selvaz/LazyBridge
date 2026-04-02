"""Unit tests for LazyAgent — mock-only, no API calls."""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from lazybridge.lazy_agent import LazyAgent
from lazybridge.core.types import (
    CompletionResponse,
    StreamChunk,
    ThinkingContent,
    ToolCall,
    ToolUseContent,
    UsageStats,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent(provider="anthropic"):
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider.get_default_max_tokens.return_value = 4096
    mock_exec.model = "test-model"
    agent._executor = mock_exec
    import uuid
    agent.id = str(uuid.uuid4())
    agent.name = "test"
    agent.description = None
    agent.system = None
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent.session = None
    agent._log = None
    return agent


def _tool_response(tool_name="calc", tool_id="t1", text="", thinking=""):
    """CompletionResponse that requests one tool call."""
    return CompletionResponse(
        content=text,
        thinking=thinking or None,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, arguments={"x": 1})],
        stop_reason="tool_use",
        usage=UsageStats(),
    )


def _final_response(content="done"):
    return CompletionResponse(content=content, usage=UsageStats())


# ── T6.01 — chat: last_output set after single call ──────────────────────────

def test_chat_sets_last_output(mock_execute, fake_response):
    agent = _make_agent()
    agent._executor.execute.return_value = fake_response
    resp = agent.chat("hello")
    assert agent._last_output == "mock response"


# ── T6.02 — loop: terminates when no tool calls returned ─────────────────────

def test_loop_terminates_no_tool_calls(fake_response):
    agent = _make_agent()
    agent._executor.execute.return_value = fake_response
    resp = agent.loop("do it")
    assert resp.content == "mock response"


# ── T6.03 — loop: max_steps=0 raises ValueError ───────────────────────────────

def test_loop_max_steps_zero():
    agent = _make_agent()
    with pytest.raises(ValueError, match="max_steps"):
        agent.loop("task", max_steps=0)


# ── T6.04 — loop: stream=True raises TypeError ───────────────────────────────

def test_loop_stream_raises():
    agent = _make_agent()
    with pytest.raises(TypeError, match="stream"):
        agent.loop("task", stream=True)


# ── T6.05 — aloop: thinking blocks included in assistant message ──────────────
#
# This is the regression test for the audit finding: aloop() was omitting
# ThinkingContent from the assistant message, which breaks the Anthropic API's
# required message format for thinking-enabled models.
#
# The fix: aloop() must mirror loop()'s thinking-block inclusion.

async def test_aloop_preserves_thinking_blocks_in_convo():
    """aloop() must include ThinkingContent in the assistant message when the
    model returns thinking alongside a tool call."""
    agent = _make_agent()

    # First response: thinking + tool call
    step1 = _tool_response(thinking="let me think")
    # Second response: final answer (no tools)
    step2 = _final_response("done")

    call_count = 0

    async def fake_aexecute(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return step1
        return step2

    agent._executor.aexecute = fake_aexecute

    # Dummy tool handler
    def tool_runner(name, args):
        return "tool result"

    await agent.aloop("do it with thinking", tool_runner=tool_runner)

    # The second call's messages must contain a ThinkingContent block in the
    # assistant turn (the first response's content was appended to convo).
    # Verify by checking the request that was passed to aexecute on step 2.
    # We do this by capturing the conversation at step 2.
    captured_requests = []

    call_count = 0

    async def recording_aexecute(request):
        nonlocal call_count
        call_count += 1
        captured_requests.append(request)
        if call_count == 1:
            return step1
        return step2

    agent._executor.aexecute = recording_aexecute
    agent._last_output = None
    call_count = 0

    await agent.aloop("do it with thinking", tool_runner=tool_runner)

    assert len(captured_requests) == 2
    # The second request must have an assistant message containing ThinkingContent
    second_request = captured_requests[1]
    assistant_messages = [m for m in second_request.messages if m.role.value == "assistant"]
    assert assistant_messages, "No assistant message found in second request"
    asst_msg = assistant_messages[0]
    content = asst_msg.content if isinstance(asst_msg.content, list) else []
    thinking_blocks = [b for b in content if isinstance(b, ThinkingContent)]
    assert thinking_blocks, (
        "aloop() dropped ThinkingContent from the assistant message. "
        "Expected at least one ThinkingContent block."
    )


# ── T6.06 — text() shorthand returns content string ──────────────────────────

def test_text_returns_content(fake_response):
    agent = _make_agent()
    agent._executor.execute.return_value = fake_response
    result = agent.text("hi")
    assert result == "mock response"


# ── T6.07 — text() with stream=True raises TypeError ─────────────────────────

def test_text_stream_raises():
    agent = _make_agent()
    with pytest.raises(TypeError, match="stream"):
        agent.text("hi", stream=True)


# ── T6.08 — json() returns parsed result and injects _JSON_SYSTEM_SUFFIX ─────

def test_json_returns_parsed_and_injects_suffix(fake_response):
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: str

    parsed_instance = Answer(value="42")
    resp = CompletionResponse(
        content='{"value": "42"}',
        parsed=parsed_instance,
        usage=UsageStats(),
    )
    agent = _make_agent()
    agent._executor.execute.return_value = resp

    captured = {}

    original_chat = agent.chat

    def capturing_chat(messages, **kwargs):
        captured["system"] = kwargs.get("system")
        return original_chat(messages, **kwargs)

    agent.chat = capturing_chat
    result = agent.json("what is 6x7?", schema=Answer)

    assert result is parsed_instance
    assert agent._JSON_SYSTEM_SUFFIX in (captured.get("system") or "")


# ── T6.09 — json() with stream=True raises TypeError ─────────────────────────

def test_json_stream_raises():
    # T6.09
    agent = _make_agent()
    with pytest.raises(TypeError, match="stream"):
        agent.json("hi", schema=dict, stream=True)


# ── T6.10 — json() preserves caller-supplied system prompt ───────────────────

def test_json_appends_suffix_to_existing_system(fake_response):
    # T6.10
    from pydantic import BaseModel

    class Out(BaseModel):
        x: int

    resp = CompletionResponse(content='{"x": 1}', parsed=Out(x=1), usage=UsageStats())
    agent = _make_agent()
    agent._executor.execute.return_value = resp

    captured = {}
    original_chat = agent.chat

    def capturing_chat(messages, **kwargs):
        captured["system"] = kwargs.get("system")
        return original_chat(messages, **kwargs)

    agent.chat = capturing_chat
    agent.json("hi", schema=Out, system="Be concise.")

    sys = captured.get("system", "")
    assert "Be concise." in sys
    assert agent._JSON_SYSTEM_SUFFIX in sys


# ── T6.11 — chat(stream=True) returns Iterator[StreamChunk] ──────────────────

def test_chat_stream_returns_iterator(fake_chunks):
    # T6.11 — _make_agent() uses a raw MagicMock executor; configure .stream directly
    agent = _make_agent()
    agent._executor.stream.return_value = iter(fake_chunks)
    gen = agent.chat("tell me a story", stream=True)
    chunks = list(gen)
    assert len(chunks) == 2
    assert chunks[0].delta == "mock "
    assert chunks[1].is_final is True


# ── T6.12 — loop() executes tool and continues until no tool_calls ───────────

def test_loop_executes_tool_and_continues():
    # T6.12
    agent = _make_agent()

    step1 = _tool_response(tool_name="adder", tool_id="t1")
    step2 = _final_response("result is 42")

    call_count = 0

    def fake_execute(request):
        nonlocal call_count
        call_count += 1
        return step1 if call_count == 1 else step2

    agent._executor.execute = fake_execute

    def my_runner(name, args):
        assert name == "adder"
        return 42

    result = agent.loop("add 1+1", tool_runner=my_runner)
    assert result.content == "result is 42"
    assert call_count == 2


# ── T6.13 — aloop() executes tool and continues ───────────────────────────────

async def test_aloop_executes_tool_and_continues():
    # T6.13
    agent = _make_agent()
    step1 = _tool_response(tool_name="calc", tool_id="t2")
    step2 = _final_response("async done")
    call_count = 0

    async def fake_aexecute(request):
        nonlocal call_count
        call_count += 1
        return step1 if call_count == 1 else step2

    agent._executor.aexecute = fake_aexecute

    def runner(name, args):
        return "tool_output"

    result = await agent.aloop("async task", tool_runner=runner)
    assert result.content == "async done"
    assert call_count == 2


# ── T6.14 — _build_effective_system: system + context + guidance ─────────────

def test_build_effective_system_all_parts():
    # T6.14
    from lazybridge.lazy_context import LazyContext
    from lazybridge.lazy_tool import NormalizedToolSet

    agent = _make_agent()
    agent.system = "base system"

    ctx = LazyContext.from_text("context text")
    toolset = NormalizedToolSet([], [], {})

    result = agent._build_effective_system("extra", ctx, toolset)
    assert "base system" in result
    assert "extra" in result
    assert "context text" in result
