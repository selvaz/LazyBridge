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
    import claude_agent_sdk
    from claude_agent_sdk._internal.transport import subprocess_cli

    killed = []

    class Transport:
        def __init__(self, **kwargs):
            self._process = SimpleNamespace(pid=12345, returncode=None)

        async def close(self):
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    continue

    async def query(*, transport, **kwargs):
        try:
            yield object()
            await asyncio.Event().wait()
        finally:
            await transport.close()

    monkeypatch.setattr(subprocess_cli, "SubprocessCLITransport", Transport)
    monkeypatch.setattr(claude_agent_sdk, "query", query)
    monkeypatch.setattr(sdk_client, "_kill_tree", lambda process: killed.append(process.pid))
    monkeypatch.setattr(sdk_client, "_CLOSE_GRACE", 0.2)
    return killed


def test_sdk_run_raises_idle_timeout(stalled_sdk):
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="went idle"):
        run_coroutine_blocking(
            lambda: sdk_client.AgentSdkClient().run("test", options=ClaudeSdkOptions(stream_idle_timeout=0.2))
        )
    assert time.monotonic() - started < 2
    assert stalled_sdk == [12345]


@pytest.mark.parametrize("idle,request_timeout", [(0.2, 5), (5, 0.2)])
def test_stall_times_out_and_kills_child(stalled_sdk, idle, request_timeout):
    engine = ClaudeCodeEngine(stream_idle_timeout=idle, request_timeout=request_timeout, max_retries=0)
    started = time.monotonic()
    result = Agent(engine=engine)("test")
    assert time.monotonic() - started < 2
    assert not result.ok
    assert result.error.type == "TimeoutError"
    assert stalled_sdk == [12345]


def test_fast_sdk_run(monkeypatch):
    import claude_agent_sdk
    from claude_agent_sdk._internal.transport import subprocess_cli

    class Transport:
        def __init__(self, **kwargs):
            self._process = None

    async def query(**kwargs):
        yield claude_agent_sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="session",
            result="done",
        )

    monkeypatch.setattr(subprocess_cli, "SubprocessCLITransport", Transport)
    monkeypatch.setattr(claude_agent_sdk, "query", query)
    result = Agent(engine=ClaudeCodeEngine(stream_idle_timeout=0.2, max_retries=0))("test")
    assert result.ok
    assert result.text() == "done"
