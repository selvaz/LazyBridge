from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from lazybridge.engines.codex.app_server import CodexAppServerClient
from lazybridge.engines.coding import ApprovalDecision

FIXTURE = str(Path(__file__).parent / "fixtures" / "fake_app_server.py")

# Bound every subprocess test so a broken fixture or protocol regression
# fails fast instead of hanging CI.
_TIMEOUT = 10.0


async def _call_tool(tool: str, arguments: dict) -> dict:
    assert tool == "get_quote"
    assert arguments == {"symbol": "AMZN"}
    return {"success": True, "contentItems": [{"type": "inputText", "text": "123.45"}]}


class _QuoteTool:
    name = "get_quote"
    description = "Return a quote"

    class _Def:
        parameters = {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}

    def definition(self):
        return self._Def()


def test_full_turn_round_trip_including_a_tool_call():
    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "happy"))
        chunks: list[str] = []

        async def on_text(chunk: str) -> None:
            chunks.append(chunk)

        result = await client.run(
            prompt="quote AMZN",
            model="gpt-5-codex",
            cwd=None,
            dynamic_tools=[
                {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
            ],
            on_tool_call=_call_tool,
            on_text=on_text,
        )
        return result, chunks

    result, chunks = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert result.text == "AMZN is 123.45"
    # Usage is the last cumulative ``total`` reported by
    # thread/tokenUsage/updated, not the first one nor the per-call ``last``.
    assert result.input_tokens == 55
    assert result.output_tokens == 7
    # The App Server reports no dollar cost under ChatGPT-plan auth.
    assert result.cost_usd == 0.0
    assert chunks == ["AMZN is ", "123.45"]


def test_no_tools_requested_skips_the_tool_round_trip():
    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "happy"))

        async def unexpected_tool_call(tool, arguments):
            raise AssertionError("no tool call was expected")

        return await client.run(
            prompt="just chat",
            model=None,
            cwd=None,
            dynamic_tools=[],
            on_tool_call=unexpected_tool_call,
        )

    result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert result.text == "AMZN is 123.45"


def test_a_server_tool_call_id_colliding_with_a_pending_request_still_works():
    """The server numbers its requests to us independently of our own.

    If ``item/tool/call`` arrives carrying an id we still have in flight
    (here: our own ``turn/start``), dispatching on the id alone would
    resolve that request's future with the tool-call params and never
    answer the tool call — the run would then hang until the engine's
    request_timeout fires.
    """

    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "id_collision"))
        return await client.run(
            prompt="quote AMZN",
            model=None,
            cwd=None,
            dynamic_tools=[
                {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
            ],
            on_tool_call=_call_tool,
        )

    result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert result.text == "AMZN is 123.45"


def test_a_terminal_error_notification_raises_and_a_retryable_one_does_not():
    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "error_notification"))
        return await client.run(prompt="quote AMZN", model=None, cwd=None, dynamic_tools=[], on_tool_call=_call_tool)

    with pytest.raises(RuntimeError, match="stream disconnected"):
        asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))


def test_a_failed_turn_status_raises_with_the_reported_message():
    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "turn_failed"))
        return await client.run(
            prompt="quote AMZN",
            model=None,
            cwd=None,
            dynamic_tools=[],
            on_tool_call=_call_tool,
        )

    with pytest.raises(RuntimeError, match="rate limited"):
        asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))


def test_developer_instructions_are_sent_on_thread_start():
    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "developer_instructions"))
        return await client.run(
            prompt="just chat",
            model=None,
            cwd=None,
            dynamic_tools=[],
            on_tool_call=_call_tool,
            developer_instructions="Be concise.",
        )

    result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert result.text == "AMZN is 123.45"


def test_server_exit_before_initialize_fails_immediately():
    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "exit_immediately"))
        return await client.run(
            prompt="just chat",
            model=None,
            cwd=None,
            dynamic_tools=[],
            on_tool_call=_call_tool,
        )

    with pytest.raises(ConnectionError, match="exited before completing"):
        asyncio.run(asyncio.wait_for(run(), timeout=2.0))


def test_native_command_approval_is_forwarded_to_the_shared_gate():
    seen = []

    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "command_approval"))

        async def gate(request):
            seen.append(request)
            return ApprovalDecision.allow_for_session()

        return await client.run(
            prompt="inspect",
            model=None,
            cwd="C:/work/project",
            dynamic_tools=[],
            on_tool_call=_call_tool,
            sandbox="workspace-write",
            approval_policy="on-request",
            approval_gate=gate,
        )

    result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert result.text == "approved"
    assert seen[0].kind == "command"
    assert seen[0].name == "git status"
