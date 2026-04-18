"""Unit tests for LazyAgent — mock-only, no API calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazybridge.core.types import (
    CompletionResponse,
    ThinkingContent,
    ToolCall,
    UsageStats,
)
from lazybridge.lazy_agent import LazyAgent

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
    agent._last_response = None
    agent.session = None
    agent.memory = None
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
    _resp = agent.chat("hello")
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
        "aloop() dropped ThinkingContent from the assistant message. Expected at least one ThinkingContent block."
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


# ── T6.09b — json() raises on structured-output validation failure ───────────


def test_json_raises_on_validation_error():
    from lazybridge.core.types import StructuredOutputParseError

    resp = CompletionResponse(
        content='{"bad": "json"}',
        parsed=None,
        validation_error="JSON parse error: expected int",
        validated=False,
        usage=UsageStats(),
    )
    agent = _make_agent()
    agent._executor.execute.return_value = resp

    with pytest.raises(StructuredOutputParseError):
        agent.json("give me a number", schema=dict)


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


# ── T2.15 — loop(verify=None) is unchanged (regression guard) ────────────────


def test_loop_verify_none_unchanged():
    """verify=None must not change loop() behaviour at all."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("answer")
    resp = agent.loop("question", verify=None)
    assert resp.content == "answer"
    assert agent._last_output == "answer"


# ── T2.16 — loop(verify=callable) approves on first attempt ─────────────────


def test_loop_verify_approves_first_attempt():
    """verify callable returns 'APPROVED' → loop exits after 1 attempt."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("correct answer")

    verify_calls = []

    def judge(question: str, answer: str) -> str:
        verify_calls.append((question, answer))
        return "APPROVED"

    resp = agent.loop("my question", verify=judge)
    assert resp.content == "correct answer"
    assert len(verify_calls) == 1
    assert verify_calls[0] == ("my question", "correct answer")


# ── T2.17 — loop(verify=callable) retries, approves on second attempt ────────


def test_loop_verify_retries_then_approves():
    """verify callable rejects once then approves — worker called twice."""
    agent = _make_agent()
    call_count = 0

    def fake_execute(request):
        nonlocal call_count
        call_count += 1
        return _final_response(f"attempt {call_count}")

    agent._executor.execute = fake_execute

    verdicts = ["RETRY: too vague", "APPROVED"]

    def judge(question: str, answer: str) -> str:
        return verdicts.pop(0)

    resp = agent.loop("question", verify=judge, max_verify=3)
    assert resp.content == "attempt 2"
    assert call_count == 2


# ── T2.18 — loop(verify=callable) returns last result when max_verify hit ────


def test_loop_verify_max_exceeded_returns_last():
    """When max_verify is exhausted, return last worker result — no exception."""
    agent = _make_agent()
    call_count = 0

    def fake_execute(request):
        nonlocal call_count
        call_count += 1
        return _final_response(f"attempt {call_count}")

    agent._executor.execute = fake_execute

    def always_reject(question: str, answer: str) -> str:
        return "RETRY: not good enough"

    resp = agent.loop("question", verify=always_reject, max_verify=2)
    assert call_count == 2
    assert resp.content == "attempt 2"


# ── T2.19 — loop(verify=LazyAgent) calls verify.text() with correct prompt ───


def test_loop_verify_lazy_agent_calls_text():
    """When verify is a LazyAgent, loop() must call verify.text() with the
    formatted prompt and use the verdict to decide whether to retry."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("final answer")

    mock_judge = MagicMock()
    mock_judge.text.return_value = "APPROVED - well done"

    resp = agent.loop("the question", verify=mock_judge)

    mock_judge.text.assert_called_once_with("Question: the question\nAnswer: final answer")
    assert resp.content == "final answer"


# ── T2.20 — aloop(verify=async callable) works correctly ─────────────────────


async def test_aloop_verify_async_callable():
    """aloop() with an async verify callable approves and returns result."""
    agent = _make_agent()

    async def fake_aexecute(request):
        return _final_response("async answer")

    agent._executor.aexecute = fake_aexecute

    verdicts = ["RETRY: incomplete", "APPROVED"]

    async def async_judge(question: str, answer: str) -> str:
        return verdicts.pop(0)

    call_count = 0
    original_aexecute = agent._executor.aexecute

    async def counting_aexecute(request):
        nonlocal call_count
        call_count += 1
        return await original_aexecute(request)

    agent._executor.aexecute = counting_aexecute

    resp = await agent.aloop("async question", verify=async_judge, max_verify=3)
    assert resp.content == "async answer"
    assert call_count == 2


# ── T2.21 — on_event("verify_rejected") fires on each rejection ───────────────


def test_loop_verify_rejected_event_fires():
    """on_event("verify_rejected") fires with attempt and verdict on each rejection."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("correct answer")
    verdicts = ["RETRY: needs more detail", "APPROVED: good enough"]

    def judge(q, a):
        return verdicts.pop(0)

    events = []

    def on_event(event_type, data):
        events.append((event_type, data))

    resp = agent.loop("question", verify=judge, max_verify=3, on_event=on_event)
    assert resp.content == "correct answer"

    rejected = [(t, d) for t, d in events if t == "verify_rejected"]
    assert len(rejected) == 1
    assert rejected[0][1]["attempt"] == 1
    assert rejected[0][1]["verdict"] == "RETRY: needs more detail"


# ── T2.22 — on_event("verify_rejected") NOT fired when approved first try ─────


def test_loop_verify_no_rejected_event_on_first_approval():
    """No verify_rejected event when the judge approves on the first attempt."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("correct answer")

    def judge(q, a):
        return "APPROVED: perfect"

    events = []
    agent.loop("question", verify=judge, on_event=lambda t, d: events.append(t))

    assert "verify_rejected" not in events


# ── T2.23 — resp.verify_log contains all rejected verdicts in order ───────────


def test_loop_verify_log_collects_rejections():
    """verify_log on the returned CompletionResponse holds every rejected verdict."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("correct answer")
    verdicts = ["RETRY: too short", "RETRY: missing sources", "APPROVED: ok"]

    def judge(q, a):
        return verdicts.pop(0)

    resp = agent.loop("question", verify=judge, max_verify=5)
    assert resp.verify_log == ["RETRY: too short", "RETRY: missing sources"]


# ── T2.24 — resp.verify_log is empty when verify=None ────────────────────────


def test_loop_verify_log_empty_when_no_verify():
    """verify_log is [] when verify is not set."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("correct answer")
    resp = agent.loop("question")
    assert resp.verify_log == []


# =============================================================================
# T_TR — tool result serialization (P1.5)
# =============================================================================


def test_tool_result_pydantic_serialised_as_json():
    """T_TR.1: Pydantic model returned by a tool is serialised with model_dump_json()."""
    from pydantic import BaseModel

    from lazybridge.lazy_agent import _serialise_tool_result

    class BlogPost(BaseModel):
        title: str
        intro: str

    post = BlogPost(title="AI News", intro="Big week for AI")
    result = _serialise_tool_result(post)
    import json

    parsed = json.loads(result)
    assert parsed["title"] == "AI News"
    assert parsed["intro"] == "Big week for AI"


def test_tool_result_dict_serialised_as_json():
    """T_TR.2: dict returned by a tool is serialised with json.dumps()."""
    import json

    from lazybridge.lazy_agent import _serialise_tool_result

    d = {"key": "value", "count": 42}
    result = _serialise_tool_result(d)
    parsed = json.loads(result)
    assert parsed == d


def test_tool_result_str_unchanged():
    """T_TR.3: str returned by a tool passes through as-is."""
    from lazybridge.lazy_agent import _serialise_tool_result

    result = _serialise_tool_result("plain text result")
    assert result == "plain text result"


# =============================================================================
# T_OS — output_schema as true default (P1.1)
# =============================================================================


def test_chat_uses_agent_level_output_schema():
    """T_OS.1: agent.chat() without call-level output_schema applies self.output_schema."""
    from pydantic import BaseModel

    class BlogPost(BaseModel):
        title: str

    agent = _make_agent()
    agent.output_schema = BlogPost

    resp = CompletionResponse(
        content='{"title": "Test"}',
        parsed=BlogPost(title="Test"),
        usage=UsageStats(),
    )
    agent._executor.execute.return_value = resp

    _result = agent.chat("write something")
    # Verify the request was built with the schema
    call_args = agent._executor.execute.call_args[0][0]
    assert call_args.structured_output is not None
    assert call_args.structured_output.schema is BlogPost


def test_chat_call_level_schema_overrides_agent_level():
    """T_OS.2: call-level output_schema overrides agent-level."""
    from pydantic import BaseModel

    class AgentSchema(BaseModel):
        a: str

    class CallSchema(BaseModel):
        b: str

    agent = _make_agent()
    agent.output_schema = AgentSchema

    resp = CompletionResponse(content="{}", usage=UsageStats())
    agent._executor.execute.return_value = resp

    agent.chat("msg", output_schema=CallSchema)
    call_args = agent._executor.execute.call_args[0][0]
    assert call_args.structured_output.schema is CallSchema


def test_chat_no_schema_no_structured_output():
    """T_OS.3: Without any output_schema, structured_output is None."""
    agent = _make_agent()
    agent.output_schema = None

    resp = CompletionResponse(content="hello", usage=UsageStats())
    agent._executor.execute.return_value = resp

    agent.chat("msg")
    call_args = agent._executor.execute.call_args[0][0]
    assert call_args.structured_output is None


# =============================================================================
# T_EV — AGENT_START / AGENT_FINISH / LOOP_STEP events (P1.2)
# =============================================================================


def test_loop_emits_agent_start_and_finish():
    """T_EV.1: loop() emits AGENT_START before and AGENT_FINISH after the loop."""
    from lazybridge.lazy_session import Event

    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("done")

    emitted = []

    def capture(event_type, **data):
        emitted.append(event_type)

    agent._log = MagicMock()
    agent._log.log = MagicMock(side_effect=capture)

    agent.loop("task")

    assert Event.AGENT_START in emitted
    assert Event.AGENT_FINISH in emitted
    # START must come before FINISH
    assert emitted.index(Event.AGENT_START) < emitted.index(Event.AGENT_FINISH)


def test_loop_emits_loop_step_for_tool_calls():
    """T_EV.2: loop() emits LOOP_STEP for each step that produces tool calls."""
    from lazybridge.lazy_session import Event

    agent = _make_agent()
    tool_resp = _tool_response()
    final_resp = _final_response("done")
    agent._executor.execute.side_effect = [tool_resp, final_resp]

    emitted = []

    def capture(event_type, **data):
        emitted.append(event_type)

    agent._log = MagicMock()
    agent._log.log = MagicMock(side_effect=capture)

    mock_tool = MagicMock()
    mock_tool.name = "calc"
    mock_tool.run = MagicMock(return_value=42)
    from lazybridge.lazy_tool import LazyTool

    agent.tools = [mock_tool]

    # Provide a tool in the registry via tools= param
    from lazybridge.lazy_tool import NormalizedToolSet

    fake_lazy_tool = MagicMock(spec=LazyTool)
    fake_lazy_tool.name = "calc"
    fake_lazy_tool.run = MagicMock(return_value=42)

    with patch.object(agent, "_build_tool_set") as mock_bts:
        ns = NormalizedToolSet.__new__(NormalizedToolSet)
        ns.definitions = []
        ns.bridges = []
        ns.registry = {"calc": fake_lazy_tool}
        mock_bts.return_value = ns
        agent.loop("task")

    assert Event.LOOP_STEP in emitted


def test_loop_no_loop_step_when_no_tool_calls():
    """T_EV.3: loop() does NOT emit LOOP_STEP when the model produces no tool calls."""
    from lazybridge.lazy_session import Event

    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("done")

    emitted = []

    def capture(event_type, **data):
        emitted.append(event_type)

    agent._log = MagicMock()
    agent._log.log = MagicMock(side_effect=capture)

    agent.loop("task")

    assert Event.LOOP_STEP not in emitted


# =============================================================================
# T_AE — async-aware on_event in aloop() (P1.4)
# =============================================================================


@pytest.mark.asyncio
async def test_aloop_awaits_async_on_event():
    """T_AE.1: aloop() with an async callback — the coroutine is awaited."""
    agent = _make_agent()
    agent._executor.aexecute = AsyncMock(return_value=_final_response("done"))

    calls = []

    async def async_handler(event_name, payload):
        calls.append(event_name)

    await agent.aloop("task", on_event=async_handler)

    assert "step" in calls
    assert "done" in calls


@pytest.mark.asyncio
async def test_aloop_sync_on_event_still_works():
    """T_AE.2: aloop() with a sync callback — works without error."""
    agent = _make_agent()
    agent._executor.aexecute = AsyncMock(return_value=_final_response("done"))

    calls = []

    def sync_handler(event_name, payload):
        calls.append(event_name)

    await agent.aloop("task", on_event=sync_handler)

    assert "step" in calls
    assert "done" in calls


# =============================================================================
# T_LR — _last_response stores full CompletionResponse (P2.1)
# =============================================================================


def test_chat_sets_last_response():
    """_last_response is set after chat()."""
    agent = _make_agent()
    resp = CompletionResponse(content="hello", usage=UsageStats())
    agent._executor.execute.return_value = resp

    agent.chat("msg")
    assert agent._last_response is resp


def test_loop_sets_last_response():
    """_last_response is set after loop()."""
    agent = _make_agent()
    agent._executor.execute.return_value = _final_response("done")

    agent.loop("task")
    assert agent._last_response is not None
    assert agent._last_response.content == "done"


@pytest.mark.asyncio
async def test_aloop_sets_last_response():
    """_last_response is set after aloop()."""
    agent = _make_agent()
    agent._executor.aexecute = AsyncMock(return_value=_final_response("async done"))

    await agent.aloop("task")
    assert agent._last_response is not None
    assert agent._last_response.content == "async done"


# =============================================================================
# T_RESULT — agent.result property
# =============================================================================


def test_result_returns_none_before_any_call():
    """T_RESULT.1: agent.result is None before the agent has been called."""
    agent = _make_agent()
    assert agent.result is None


def test_result_returns_text_after_chat_without_schema():
    """T_RESULT.2: agent.result returns plain text when no output_schema is active."""
    agent = _make_agent()
    resp = CompletionResponse(content="hello world", usage=UsageStats())
    agent._executor.execute.return_value = resp

    agent.chat("msg")
    assert agent.result == "hello world"
    assert isinstance(agent.result, str)


def test_result_returns_pydantic_when_output_schema_active():
    """T_RESULT.3: agent.result returns the Pydantic object when output_schema is set."""
    from pydantic import BaseModel

    class BlogPost(BaseModel):
        title: str
        intro: str

    agent = _make_agent()
    agent.output_schema = BlogPost
    post = BlogPost(title="AI in 2025", intro="It was a big year.")
    resp = CompletionResponse(
        content=post.model_dump_json(),
        parsed=post,
        usage=UsageStats(),
    )
    agent._executor.execute.return_value = resp

    agent.chat("write something")
    result = agent.result
    assert isinstance(result, BlogPost)
    assert result.title == "AI in 2025"
    assert result.intro == "It was a big year."


# ── messages_to_str edge cases ────────────────────────────────────────────


def test_messages_to_str_uses_role_enum(mock_execute):
    """_messages_to_str correctly matches Role.USER enum values."""
    from lazybridge.core.types import Message, Role
    from lazybridge.lazy_agent import _messages_to_str

    # Message with Role enum
    msgs = [Message(role=Role.USER, content="hello")]
    assert _messages_to_str(msgs) == "hello"

    # No user role — falls back to str(messages)
    msgs_no_user = [Message(role=Role.ASSISTANT, content="hi")]
    result = _messages_to_str(msgs_no_user)
    assert isinstance(result, str)


# ── chat() with context= override ────────────────────────────────────────


@patch("lazybridge.core.executor.Executor.execute")
@patch("lazybridge.core.providers.anthropic.AnthropicProvider._init_client")
def test_chat_context_override(mock_init, mock_exec):
    """Call-level context= overrides agent-level context."""
    from lazybridge.lazy_context import LazyContext

    mock_exec.return_value = CompletionResponse(content="ok", usage=UsageStats())

    agent_ctx = LazyContext.from_text("agent-level context")
    call_ctx = LazyContext.from_text("call-level context")
    agent = LazyAgent("anthropic", system="base", context=agent_ctx)

    agent.chat("hello", context=call_ctx)

    # The system prompt passed to the executor should include the call-level context
    call_args = mock_exec.call_args
    request = call_args[0][0]
    assert "call-level context" in request.system
    assert "agent-level context" not in request.system


# ── tool_choice forwarding ────────────────────────────────────────────────


@patch("lazybridge.core.executor.Executor.execute")
@patch("lazybridge.core.providers.anthropic.AnthropicProvider._init_client")
def test_tool_choice_forwarded_to_request(mock_init, mock_exec):
    """tool_choice parameter is forwarded to CompletionRequest."""
    mock_exec.return_value = CompletionResponse(content="ok", usage=UsageStats())

    agent = LazyAgent("anthropic")
    agent.chat("hello", tool_choice="required")

    call_args = mock_exec.call_args
    request = call_args[0][0]
    assert request.tool_choice == "required"


# ── tool_choice on as_tool() ────────────────────────────────────────────


def test_as_tool_tool_choice_forwarded():
    """as_tool(tool_choice=) forwards to the inner loop() call."""
    agent = _make_agent()
    # Give the agent a tool so loop() is triggered (not chat())
    dummy_tool = MagicMock(spec=["name", "definition", "guidance", "run", "arun"])
    dummy_tool.name = "dummy"
    dummy_tool.guidance = None
    dummy_tool.definition = MagicMock()
    dummy_tool.definition.name = "dummy"
    agent.tools = [dummy_tool]

    tool = agent.as_tool("test_agent", "test desc", tool_choice="required")
    assert tool._delegate is not None
    assert tool._delegate.tool_choice == "required"


# ── tool_choice="parallel" ──────────────────────────────────────────────


def test_loop_parallel_tool_calls_sequential():
    """tool_choice="parallel" in sync loop executes all tools (sequentially)."""
    agent = _make_agent()

    # Step 1: model returns 2 tool calls
    step1 = CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(id="t1", name="a", arguments={"x": 1}),
            ToolCall(id="t2", name="b", arguments={"x": 2}),
        ],
        stop_reason="tool_use",
        usage=UsageStats(),
    )
    # Step 2: final response
    step2 = CompletionResponse(content="done with both", usage=UsageStats())

    call_count = 0

    def fake_execute(request):
        nonlocal call_count
        call_count += 1
        return step1 if call_count == 1 else step2

    agent._executor.execute = MagicMock(side_effect=fake_execute)

    tool_calls_received = []

    def tool_runner(name, args):
        tool_calls_received.append(name)
        return f"result_{name}"

    resp = agent.loop("do both", tool_runner=tool_runner, tool_choice="parallel")
    assert resp.content == "done with both"
    assert "a" in tool_calls_received
    assert "b" in tool_calls_received


async def test_aloop_parallel_tool_calls():
    """tool_choice="parallel" in async loop runs tools concurrently via gather."""
    agent = _make_agent()

    step1 = CompletionResponse(
        content="",
        tool_calls=[
            ToolCall(id="t1", name="a", arguments={"x": 1}),
            ToolCall(id="t2", name="b", arguments={"x": 2}),
        ],
        stop_reason="tool_use",
        usage=UsageStats(),
    )
    step2 = CompletionResponse(content="parallel done", usage=UsageStats())

    call_count = 0

    async def fake_aexecute(request):
        nonlocal call_count
        call_count += 1
        return step1 if call_count == 1 else step2

    agent._executor.aexecute = fake_aexecute

    tool_calls_received = []

    def tool_runner(name, args):
        tool_calls_received.append(name)
        return f"result_{name}"

    resp = await agent.aloop("do both", tool_runner=tool_runner, tool_choice="parallel")
    assert resp.content == "parallel done"
    assert set(tool_calls_received) == {"a", "b"}
