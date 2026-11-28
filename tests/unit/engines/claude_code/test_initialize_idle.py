"""Initialization and idle deadlines through the SDK transport boundary."""

import asyncio
import json

import pytest

from lazybridge import Agent
from lazybridge.engines._tool_clock import ToolClock, wait_with_tool_clock
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions
from lazybridge.engines.claude_code.sdk_client import _INITIALIZED, AgentSdkClient, _bounded_query
from lazybridge.engines.coding import ApprovalDecision

pytest.importorskip("claude_agent_sdk")


class DelayedTransport:
    def __init__(self, *, prompt, options, initialize_delay, answer):
        self.initialize_delay = initialize_delay
        self.answer = answer
        self.requests = asyncio.Queue()
        self.closed = False

    async def connect(self):
        pass

    async def write(self, data):
        await self.requests.put(json.loads(data))

    async def read_messages(self):
        request = await self.requests.get()
        assert request["request"]["subtype"] == "initialize"
        if self.initialize_delay is None:
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(self.initialize_delay)
        yield {"type": "control_response", "response": {"subtype": "success", "request_id": request["request_id"]}}
        await self.requests.get()
        if not self.answer:
            await asyncio.Event().wait()
        yield {
            "type": "result",
            "subtype": "success",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "is_error": False,
            "num_turns": 1,
            "session_id": "test",
            "result": "done",
        }

    async def end_input(self):
        pass

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario", ["slow_initialize", "never_initialize", "idle_after_initialize"])
@pytest.mark.asyncio
async def test_initialize_and_idle_deadlines(stream, scenario):
    transports = []

    def factory(**kwargs):
        transport = DelayedTransport(
            **kwargs,
            initialize_delay={"slow_initialize": 0.15, "never_initialize": None, "idle_after_initialize": 0}[scenario],
            answer=scenario == "slow_initialize",
        )
        transports.append(transport)
        return transport

    agent = Agent(
        ClaudeCodeEngine(
            client=AgentSdkClient(transport_factory=factory),
            initialize_timeout=0.5,
            stream_idle_timeout=0.05,
            request_timeout=2,
            max_retries=0,
        )
    )
    if scenario == "slow_initialize":
        if stream:
            assert "".join([chunk async for chunk in agent.stream("test")]) == "done"
        else:
            result = await agent.run("test")
            assert result.ok, result.error
            assert result.text() == "done"
    else:
        expected = "Control request timeout: initialize" if scenario == "never_initialize" else "went idle"
        if stream:
            with pytest.raises(Exception, match=expected):
                _ = [chunk async for chunk in agent.stream("test")]
        else:
            result = await agent.run("test")
            assert not result.ok
            assert expected in result.error.message
    assert transports[0].closed


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_approval_waits_pause_sdk_idle_clock_and_resume_after_last_decision(raises):
    clock = ToolClock()
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]

    async def gate(request):
        index = request.arguments["index"]
        entered[index].set()
        await release[index].wait()
        try:
            if raises:
                raise ValueError("approval failed")
            return ApprovalDecision.allow()
        finally:
            finished[index].set()

    sdk_options = AgentSdkClient._sdk_options(ClaudeSdkOptions(approval_gate=gate, tool_clock=clock))

    async def query(prompt, options, transport, initialize_timeout):
        yield _INITIALIZED
        await asyncio.gather(
            *(options.can_use_tool("Bash", {"index": index}, object()) for index in range(2)),
            return_exceptions=True,
        )
        await asyncio.Event().wait()

    async def consume():
        return [item async for item in _bounded_query("test", sdk_options, 1, 0.05, clock, lambda **_: object(), query)]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 1)
        await asyncio.sleep(0.12)
        assert not task.done()
        release[0].set()
        await asyncio.wait_for(finished[0].wait(), 1)
        await asyncio.sleep(0.12)
        assert not task.done()
        release[1].set()
        await asyncio.wait_for(finished[1].wait(), 1)
        await asyncio.sleep(0.02)
        assert not task.done()
        with pytest.raises(TimeoutError, match="stream went idle"):
            await asyncio.wait_for(task, 1)
        assert not clock.idle_clock.active
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_approval_wait_does_not_pause_request_budget():
    clock = ToolClock()
    entered = asyncio.Event()

    async def gate(request):
        entered.set()
        await asyncio.Event().wait()

    callback = AgentSdkClient._sdk_options(ClaudeSdkOptions(approval_gate=gate, tool_clock=clock)).can_use_tool
    with pytest.raises(TimeoutError):
        await wait_with_tool_clock(callback("Bash", {}, object()), 0.05, clock)
    assert entered.is_set()
    assert not clock.idle_clock.active
