"""Initialization and idle deadlines through the SDK transport boundary."""

import asyncio
import json

import pytest

from lazybridge import Agent
from lazybridge.engines.claude_code import ClaudeCodeEngine
from lazybridge.engines.claude_code.sdk_client import AgentSdkClient

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
