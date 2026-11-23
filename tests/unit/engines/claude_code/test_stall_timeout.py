from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from lazybridge import Agent
from lazybridge._asyncbridge import run_coroutine_blocking
from lazybridge.engines.claude_code import ClaudeCodeEngine, sdk_client
from lazybridge.engines.claude_code.protocol import ClaudeSdkOptions

pytest.importorskip("claude_agent_sdk")


@pytest.fixture
def stalled_sdk(monkeypatch):
    killed = []

    class Transport:
        def __init__(self, **kwargs):
            self._process = SimpleNamespace(pid=12345, returncode=None)

        async def connect(self):
            pass

        async def read_messages(self):
            if False:
                yield {}

        async def write(self, data):
            pass

        async def close(self):
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    continue

    async def query(prompt, options, transport, initialize_timeout):
        try:
            yield object()
            await asyncio.Event().wait()
        finally:
            await transport.close()

    monkeypatch.setattr(sdk_client, "_kill_tree", lambda process: killed.append(process.pid))
    monkeypatch.setattr(sdk_client, "_CLOSE_GRACE", 0.2)
    return killed, sdk_client.AgentSdkClient(transport_factory=Transport, query_factory=query)


def test_sdk_run_raises_idle_timeout(stalled_sdk):
    killed, client = stalled_sdk
    started = time.monotonic()
    with pytest.raises(TimeoutError, match=r"ClaudeCodeEngine\(stream_idle_timeout=\.\.\.\)"):
        run_coroutine_blocking(lambda: client.run("test", options=ClaudeSdkOptions(stream_idle_timeout=0.2)))
    assert time.monotonic() - started < 2
    assert killed == [12345]


def test_custom_query_without_initialized_sentinel_starts_idle_guard_at_first_message(monkeypatch):
    class Transport:
        def __init__(self, **kwargs):
            self._process = None

        async def close(self):
            pass

    async def query(prompt, options, transport, initialize_timeout):
        # A custom factory that omits _INITIALIZED keeps its first yield in the
        # initialization phase, where the message-idle watchdog is disabled.
        await asyncio.sleep(0.1)
        yield object()
        await asyncio.Event().wait()

    monkeypatch.setattr(sdk_client, "_CLOSE_GRACE", 0.01)
    client = sdk_client.AgentSdkClient(transport_factory=Transport, query_factory=query)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="went idle"):
        run_coroutine_blocking(lambda: client.run("test", options=ClaudeSdkOptions(stream_idle_timeout=0.05)))
    assert time.monotonic() - started >= 0.1


@pytest.mark.parametrize("idle,request_timeout", [(0.2, 5), (5, 0.2)])
def test_stall_times_out_and_kills_child(stalled_sdk, idle, request_timeout):
    killed, client = stalled_sdk
    engine = ClaudeCodeEngine(client=client, stream_idle_timeout=idle, request_timeout=request_timeout, max_retries=0)
    started = time.monotonic()
    result = Agent(engine=engine)("test")
    assert time.monotonic() - started < 2
    assert not result.ok
    assert result.error.type == "TimeoutError"
    assert killed == [12345]


def test_fast_sdk_run(monkeypatch):
    import claude_agent_sdk

    class Transport:
        def __init__(self, **kwargs):
            self._process = None

        async def connect(self):
            pass

        async def read_messages(self):
            if False:
                yield {}

        async def write(self, data):
            pass

        async def close(self):
            pass

    async def query(prompt, options, transport, initialize_timeout):
        yield claude_agent_sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
            result="done",
        )

    client = sdk_client.AgentSdkClient(transport_factory=Transport, query_factory=query)
    result = Agent(engine=ClaudeCodeEngine(client=client, stream_idle_timeout=0.2, max_retries=0))("test")
    assert result.ok
    assert result.text() == "done"


def test_request_timeout_during_idle_cleanup_still_kills(stalled_sdk):
    # The overall timeout fires while the idle-timeout cleanup is in its grace
    # period (0.05 idle + 0.2 grace > 0.1 request): the child must still be
    # killed and the synchronous caller must return promptly.
    killed, client = stalled_sdk
    engine = ClaudeCodeEngine(client=client, stream_idle_timeout=0.05, request_timeout=0.1, max_retries=0)
    started = time.monotonic()
    result = Agent(engine=engine)("test")
    assert time.monotonic() - started < 2
    assert not result.ok
    assert result.error.type == "TimeoutError"
    assert 12345 in killed


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_sdk_silence_during_application_tool_does_not_timeout(monkeypatch, stream):
    import claude_agent_sdk

    class Transport:
        def __init__(self, **kwargs):
            self._process = None

        async def connect(self):
            pass

        async def read_messages(self):
            if False:
                yield {}

        async def write(self, data):
            pass

        async def close(self):
            pass

    async def slow_tool() -> str:
        """Simulate a sub-agent."""
        await asyncio.sleep(0.8)
        return "done"

    async def query(prompt, options, transport, initialize_timeout):
        # Silence before and after the tool also checks that the idle window
        # restarts, rather than merely subtracting the tool duration.
        await asyncio.sleep(0.2)
        await options.mcp_tools[0].handler({})
        await asyncio.sleep(0.2)
        yield claude_agent_sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
            result="done",
        )

    monkeypatch.setattr(sdk_client.AgentSdkClient, "_sdk_options", staticmethod(lambda options: options))
    client = sdk_client.AgentSdkClient(transport_factory=Transport, query_factory=query)
    engine = ClaudeCodeEngine(client=client, stream_idle_timeout=0.3, request_timeout=2, max_retries=0)
    agent = Agent(engine=engine, tools=[slow_tool])
    if stream:
        assert "".join([chunk async for chunk in agent.stream("delegate")]) == "done"
    else:
        result = await agent.run("delegate")
        assert result.ok, result.error
        assert result.text() == "done"
