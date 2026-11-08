"""Application tools suspend coding-engine deadlines, including SDK silence."""

import asyncio

import pytest

from lazybridge import Agent, Tool
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkStreamEvent

from .test_engine import FakeSdk


async def slow_tool(symbol: str) -> str:
    """Simulate a delegated sub-agent."""
    await asyncio.sleep(0.8)
    return symbol


class ToolSdk(FakeSdk):
    async def stream(self, prompt, *, options, attachments=()):
        result = await self.run(prompt, options=options, attachments=attachments)
        yield ClaudeSdkStreamEvent(text=result.text, final=True)


def test_request_timeout_excludes_application_tool_time():
    sdk = ToolSdk()
    agent = Agent(
        engine=ClaudeCodeEngine(client=sdk, request_timeout=0.3, max_retries=0),
        tools=[Tool.wrap(slow_tool, name="get_quote")],
    )
    result = agent("delegate")
    assert result.ok, result.error
    assert result.text() == "AMZN is available"
    assert not sdk.options[-1].tool_clock.active


@pytest.mark.asyncio
async def test_stream_idle_timeout_excludes_application_tool_time():
    sdk = ToolSdk()
    agent = Agent(
        engine=ClaudeCodeEngine(client=sdk, stream_idle_timeout=0.3, max_retries=0),
        tools=[Tool.wrap(slow_tool, name="get_quote")],
    )
    assert "".join([chunk async for chunk in agent.stream("delegate")]) == "AMZN is available"
    assert not sdk.options[-1].tool_clock.active


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_tool_error_releases_clock_and_runtime_stall_times_out(stream):
    async def broken_tool(symbol: str) -> str:
        """Fail after spending longer than the engine budget in a tool."""
        await asyncio.sleep(0.2)
        raise ValueError("tool failed")

    class StallingSdk(ToolSdk):
        async def run(self, prompt, *, options, attachments=()):
            self.options.append(options)
            response = await options.mcp_tools[0].handler({"symbol": "AMZN"})
            assert response["isError"]
            assert not options.tool_clock.active
            await asyncio.sleep(10)

    sdk = StallingSdk()
    agent = Agent(
        engine=ClaudeCodeEngine(client=sdk, request_timeout=0.1, stream_idle_timeout=0.1, max_retries=0),
        tools=[broken_tool],
    )
    if stream:
        with pytest.raises(TimeoutError, match="Claude Code stream went idle"):
            _ = [chunk async for chunk in agent.stream("fail")]
    else:
        result = await agent.run("fail")
        assert not result.ok
        assert result.error.type == "TimeoutError"
        assert result.error.message == ""
    assert not sdk.options[-1].tool_clock.active


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_cancellation_releases_application_tool_clock(stream):
    entered = asyncio.Event()

    async def hanging_tool(symbol: str) -> str:
        """Wait for cancellation."""
        entered.set()
        await asyncio.sleep(10)
        return symbol

    sdk = ToolSdk()
    agent = Agent(engine=ClaudeCodeEngine(client=sdk), tools=[Tool.wrap(hanging_tool, name="get_quote")])

    async def call():
        if stream:
            return [chunk async for chunk in agent.stream("cancel")]
        return await agent.run("cancel")

    task = asyncio.create_task(call())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not sdk.options[-1].tool_clock.active
