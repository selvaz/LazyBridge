from __future__ import annotations

import asyncio

from lazybridge import Agent, Envelope, Memory, Session, Tool
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions, ClaudeSdkResult, ClaudeSdkStreamEvent


class FakeSdk:
    def __init__(self):
        self.options: list[ClaudeSdkOptions] = []
        self.prompts: list[str] = []

    async def run(self, prompt: str, *, options: ClaudeSdkOptions) -> ClaudeSdkResult:
        self.options.append(options)
        self.prompts.append(prompt)
        assert options.builtin_tools == ("WebSearch", "WebFetch")
        assert len(options.mcp_tools) == 1
        tool = options.mcp_tools[0]
        arguments = {"symbol": "AMZN"} if tool.name == "get_quote" else {"task": "analyse AMZN"}
        tool_result = await tool.handler(arguments)
        assert tool_result["content"]
        return ClaudeSdkResult(text="AMZN is available", session_id="fake-session", input_tokens=11, output_tokens=7)

    async def stream(self, prompt: str, *, options: ClaudeSdkOptions):
        self.options.append(options)
        self.prompts.append(prompt)
        yield ClaudeSdkStreamEvent(text="streamed ")
        yield ClaudeSdkStreamEvent(text="answer")
        yield ClaudeSdkStreamEvent(session_id="fake-stream-session", final=True)


def get_quote(symbol: str) -> dict[str, str]:
    """Return a deterministic quote lookup."""
    return {"symbol": symbol}


def test_engine_works_through_a_standard_agent_and_updates_memory():
    memory = Memory()
    session = Session()
    agent = Agent(
        name="claude-code-prototype",
        engine=ClaudeCodeEngine(client=FakeSdk()),
        # A plain callable is the normal, zero-ceremony LazyBridge input.
        # Agent normalises it before the engine receives its tool list.
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    result = agent("Find AMZN")

    assert result.ok
    assert result.text() == "AMZN is available"
    assert "Find AMZN" in memory.text()
    assert result.metadata.provider == "claude-code"


def test_engine_streams_incremental_chunks_and_records_memory():
    memory = Memory()
    agent = Agent(
        name="streaming",
        engine=ClaudeCodeEngine(client=FakeSdk()),
        tools=[Tool.wrap(get_quote, name="get_quote")],
        memory=memory,
    )

    async def collect() -> str:
        return "".join([chunk async for chunk in agent.stream("Stream this")])

    assert asyncio.run(collect()) == "streamed answer"
    assert "Stream this" in memory.text()


def test_runtime_session_is_kept_on_lazybridge_session_and_skips_parent_memory():
    client = FakeSdk()
    memory = Memory()
    session = Session()
    agent = Agent(
        name="persistent",
        engine=ClaudeCodeEngine(client=client, session_mode="runtime", session_name="fixed-name"),
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    assert agent("First turn").ok
    assert agent("Second turn").ok

    assert client.options[0].resume is None
    assert client.options[1].resume == "fake-session"
    assert "First turn" not in client.prompts[1]
    assert memory.text()  # still available to explicit from_memory users


def test_runtime_session_streaming_resumes_without_reinjecting_memory():
    client = FakeSdk()
    memory = Memory()
    session = Session()
    agent = Agent(
        name="streaming-persistent",
        engine=ClaudeCodeEngine(client=client, session_mode="runtime"),
        tools=[get_quote],
        memory=memory,
        session=session,
    )

    async def collect(task: str) -> str:
        return "".join([chunk async for chunk in agent.stream(task)])

    assert asyncio.run(collect("First stream")) == "streamed answer"
    assert asyncio.run(collect("Second stream")) == "streamed answer"
    assert client.options[-2].resume is None
    assert client.options[-1].resume == "fake-stream-session"
    assert "First stream" not in client.prompts[-1]


class ChildEngine:
    async def run(self, env, *, tools, output_type, memory, session, store=None, plan_state=None):
        return Envelope(task=env.task, payload="specialist result")

    async def stream(self, env, *, tools, output_type, memory, session):
        yield "specialist result"


def test_another_lazybridge_agent_is_exposed_as_a_normal_mcp_tool():
    specialist = Agent(name="quote_specialist", engine=ChildEngine())
    parent = Agent(name="parent", engine=ClaudeCodeEngine(client=FakeSdk()), tools=[specialist])

    assert parent("Delegate quote analysis").ok


def test_standard_tool_provider_is_expanded_before_the_engine_runs():
    class QuoteProvider:
        _is_lazy_tool_provider = True

        def as_tools(self):
            return [Tool.wrap(get_quote, name="get_quote")]

    agent = Agent(
        name="provider-parent",
        engine=ClaudeCodeEngine(client=FakeSdk()),
        tools=[QuoteProvider()],
    )

    assert agent("Find AMZN through the provider").ok
