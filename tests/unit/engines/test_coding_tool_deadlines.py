"""Shared contracts for per-run budgets and concurrent application tools."""

import asyncio

import pytest

from lazybridge import Agent, Session
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkResult
from lazybridge.engines.codex.app_server import CodexRunResult
from lazybridge.engines.codex.engine import CodexEngine


class Runtime:
    """SDK/App Server fake that can invoke application tools or use native time."""

    def __init__(self, behavior):
        self.behavior = behavior

    async def run(self, prompt="", *, options=None, on_tool_call=None, dynamic_tools=(), progress=None, **kwargs):
        if progress is not None and (activity := progress.get("on_activity")):
            activity()

        async def tool(arguments):
            if options is not None:
                return await options.mcp_tools[0].handler(arguments)
            return await on_tool_call(dynamic_tools[0]["name"], arguments)

        text = await self.behavior(prompt, tool, options)
        return ClaudeSdkResult(text=text) if options is not None else CodexRunResult(text=text)


@pytest.mark.parametrize("engine_type", [ClaudeCodeEngine, CodexEngine])
@pytest.mark.asyncio
async def test_overlapping_tools_keep_budget_paused_until_last_finishes(engine_type):
    entered = asyncio.Event()

    async def delegate(delay: float) -> str:
        """Simulate parallel sub-agents."""
        entered.set()
        await asyncio.sleep(delay)
        return "done"

    async def behavior(prompt, tool, options):
        first = asyncio.create_task(tool({"delay": 0.4}))
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(tool({"delay": 0.8}))
        await asyncio.gather(first, second)
        return "done"

    agent = Agent(
        engine=engine_type(client=Runtime(behavior), request_timeout=0.3, max_retries=0),
        tools=[delegate],
    )
    result = await asyncio.wait_for(agent.run("parallel"), 3)
    assert result.ok, result.error


@pytest.mark.parametrize("engine_type", [ClaudeCodeEngine, CodexEngine])
@pytest.mark.asyncio
async def test_one_run_tools_do_not_pause_another_run_on_shared_engine(engine_type):
    entered = asyncio.Event()

    async def delegate() -> str:
        """Simulate a sub-agent."""
        entered.set()
        await asyncio.sleep(0.6)
        return "done"

    async def behavior(prompt, tool, options):
        if "delegate" in prompt:
            await tool({})
            return "done"
        await asyncio.sleep(0.5)
        return "should time out"

    engine = engine_type(client=Runtime(behavior), request_timeout=0.3, max_retries=0)
    agent = Agent(engine=engine, tools=[delegate])
    delegated = asyncio.create_task(agent.run("delegate"))
    await asyncio.wait_for(entered.wait(), 2)
    stalled = await agent.run("stall")
    assert stalled.error.type == "TimeoutError"
    assert (await delegated).ok


@pytest.mark.parametrize("engine_type", [ClaudeCodeEngine, CodexEngine])
@pytest.mark.asyncio
async def test_native_tool_time_still_counts(engine_type):
    async def behavior(prompt, tool, options):
        if options is not None:
            options.tool_observer("call", {"tool_name": "Bash", "arguments": {}, "native": True})
        # Native tools stay inside the runtime. They never invoke tool().
        await asyncio.sleep(0.5)
        return "native result"

    agent = Agent(
        engine=engine_type(client=Runtime(behavior), request_timeout=0.1, max_retries=0),
        session=Session(),
    )
    result = await agent.run("native tool")
    assert result.error.type == "TimeoutError"
