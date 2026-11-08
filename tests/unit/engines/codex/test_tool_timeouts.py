"""Dynamic tools suspend coding-engine deadlines without hiding runtime stalls."""

import asyncio

import pytest

from lazybridge import Agent, Tool
from lazybridge.engines._tool_clock import ToolClock
from lazybridge.engines.codex import engine as engine_module
from lazybridge.engines.codex.engine import CodexEngine

from .test_engine import FakeAppServer


async def slow_tool(symbol: str) -> str:
    """Simulate a delegated sub-agent."""
    await asyncio.sleep(0.8)
    return symbol


class ToolAppServer(FakeAppServer):
    async def run(self, *, progress=None, **kwargs):
        if activity := progress.get("on_activity"):
            activity()
        return await super().run(progress=progress, **kwargs)


@pytest.mark.parametrize("deadline", ["request_timeout", "stream_idle_timeout"])
@pytest.mark.asyncio
async def test_deadline_excludes_application_tool_time(deadline):
    fake = ToolAppServer()
    agent = Agent(engine=CodexEngine(client=fake, max_retries=0, **{deadline: 0.3}), tools=[slow_tool])
    result = await asyncio.wait_for(agent.run("delegate"), 3)
    assert result.ok, result.error
    assert fake.tool_results[0]["success"]


@pytest.mark.asyncio
async def test_stream_idle_timeout_excludes_application_tool_time():
    agent = Agent(
        engine=CodexEngine(client=ToolAppServer(), stream_idle_timeout=0.3, max_retries=0),
        tools=[slow_tool],
    )
    assert "".join([chunk async for chunk in agent.stream("delegate")]) == "streamed answer"


@pytest.mark.parametrize("deadline", ["request_timeout", "stream_idle_timeout"])
@pytest.mark.asyncio
async def test_runtime_stall_without_tool_still_times_out(deadline):
    stopped = asyncio.Event()

    class StallingServer(ToolAppServer):
        async def run(self, *, progress=None, **kwargs):
            if activity := progress.get("on_activity"):
                activity()
            try:
                await asyncio.sleep(10)
            finally:
                stopped.set()

    agent = Agent(engine=CodexEngine(client=StallingServer(), max_retries=0, **{deadline: 0.1}))
    result = await asyncio.wait_for(agent.run("stall"), 2)
    assert result.error.type == "TimeoutError"
    if deadline == "request_timeout":
        assert result.error.message == ""
    else:
        assert result.error.message == (
            "Codex turn went idle for 0.1s without an event "
            "(set CodexEngine(stream_idle_timeout=...) to a higher value if needed)."
        )
    assert stopped.is_set()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_tool_error_releases_clock_and_runtime_stall_times_out(monkeypatch, stream):
    clock = ToolClock()
    monkeypatch.setattr(engine_module, "ToolClock", lambda: clock)

    async def broken_tool(symbol: str) -> str:
        """Fail after spending longer than the engine budget in a tool."""
        await asyncio.sleep(0.2)
        raise ValueError("tool failed")

    class StallingServer(ToolAppServer):
        async def run(self, *, on_tool_call, progress=None, **kwargs):
            if activity := progress.get("on_activity"):
                activity()
            response = await on_tool_call("broken_tool", {"symbol": "AMZN"})
            assert not response["success"]
            assert not clock.active
            await asyncio.sleep(10)

    agent = Agent(
        engine=CodexEngine(client=StallingServer(), request_timeout=0.1, stream_idle_timeout=0.1, max_retries=0),
        tools=[broken_tool],
    )
    if stream:
        with pytest.raises(TimeoutError, match="Codex stream went idle"):
            _ = [chunk async for chunk in agent.stream("fail")]
    else:
        result = await asyncio.wait_for(agent.run("fail"), 2)
        assert result.error.type == "TimeoutError"
    assert not clock.active


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_cancellation_releases_application_tool_clock(monkeypatch, stream):
    clock = ToolClock()
    monkeypatch.setattr(engine_module, "ToolClock", lambda: clock)
    entered = asyncio.Event()

    async def hanging_tool(symbol: str) -> str:
        """Wait for cancellation."""
        entered.set()
        await asyncio.sleep(10)
        return symbol

    agent = Agent(engine=CodexEngine(client=ToolAppServer()), tools=[Tool(hanging_tool)])

    async def call():
        if stream:
            return [chunk async for chunk in agent.stream("cancel")]
        return await agent.run("cancel")

    task = asyncio.create_task(call())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not clock.active


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_idle_window_restarts_after_last_tool(stream):
    class QuietServer(ToolAppServer):
        async def run(self, *, progress=None, **kwargs):
            if activity := progress.get("on_activity"):
                activity()
            await asyncio.sleep(0.2)
            result = await super().run(progress=progress, **kwargs)
            await asyncio.sleep(0.2)
            return result

    # For streaming, FakeAppServer sends text after the tool. Keep the sink
    # silent throughout the full interval to exercise its idle guard as well.
    class QuietStreamingServer(QuietServer):
        async def run(self, *, on_text=None, **kwargs):
            return await super().run(on_text=None, **kwargs)

    agent = Agent(
        engine=CodexEngine(client=QuietStreamingServer(), stream_idle_timeout=0.3, max_retries=0),
        tools=[slow_tool],
    )
    if stream:
        assert "".join([chunk async for chunk in agent.stream("delegate")]) == "AMZN is available"
    else:
        result = await asyncio.wait_for(agent.run("delegate"), 3)
        assert result.ok, result.error
