from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from lazybridge.engines.codex.app_server import (
    CodexAppServerClient,
    CodexRequestRejected,
    CodexTurnUncertain,
)
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


def test_a_message_larger_than_the_stream_reader_default_is_read():
    """A single JSON-RPC line over ``StreamReader``'s 64 KiB default limit.

    Regression: the App Server puts whole file contents and command output
    (a real ``git diff``) in one notification, so this is routine traffic —
    without an explicit ``limit=`` on the subprocess, ``readline()`` raised
    ``ValueError: Separator is found, but chunk is longer than limit`` and the
    turn died with "Codex App Server reader failed" (hit live on the first
    diff-scoped code review).
    """
    chunks: list[str] = []

    async def on_text(chunk: str) -> None:
        chunks.append(chunk)

    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "huge_message"))
        return await client.run(
            prompt="read a big file",
            model=None,
            cwd=None,
            dynamic_tools=[],
            on_tool_call=_call_tool,
            on_text=on_text,
        )

    result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert result.text == "big"
    assert len(chunks) == 1 and len(chunks[0]) == 256 * 1024


class TestDurableThreads:
    """``thread/resume``: Codex keeps the transcript, the caller keeps an id.

    Verified live against codex-cli 0.148.0 before being fixtured here: a
    thread started with ``ephemeral=False`` resumes from a *different*
    subprocess and still knows what the first turn said, while an ephemeral
    one answers ``no rollout found for thread id``.
    """

    @staticmethod
    async def _run(scenario: str, **kwargs):
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, scenario))
        return await client.run(
            prompt="carry on",
            model=None,
            cwd="C:/work/project",
            dynamic_tools=[
                {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
            ],
            on_tool_call=_call_tool,
            **kwargs,
        )

    def test_resuming_returns_the_thread_answer_and_id(self):
        result = asyncio.run(asyncio.wait_for(self._run("resume", thread_id="thread-1"), timeout=_TIMEOUT))

        assert result.text == "resumed answer"
        assert result.thread_id == "thread-1"

    def test_usage_is_this_turn_not_the_thread_total(self):
        # The fixture replays 100/20 already spent, then reports 155/27
        # cumulative — this turn cost 55/7, and reporting 155/27 would inflate
        # every turn by the whole history before it.
        result = asyncio.run(asyncio.wait_for(self._run("resume", thread_id="thread-1"), timeout=_TIMEOUT))

        assert (result.input_tokens, result.output_tokens) == (55, 7)

    def test_a_completion_for_another_turn_is_ignored(self):
        result = asyncio.run(
            asyncio.wait_for(self._run("resume_stale_turn", thread_id="thread-1"), timeout=_TIMEOUT)
        )

        assert result.text == "resumed answer"  # not "STALE"

    def test_a_turn_lost_after_acceptance_is_not_retryable(self):
        # turn/start was acknowledged, then the server died: the turn may
        # already be committed to the durable rollout, so replaying it would
        # duplicate it (and any tool side effect).
        with pytest.raises(CodexTurnUncertain) as excinfo:
            asyncio.run(asyncio.wait_for(self._run("resume_dies_mid_turn", thread_id="thread-1"), timeout=_TIMEOUT))

        assert excinfo.value.thread_id == "thread-1"
        assert excinfo.value.turn_id == "turn-2"
        assert not isinstance(excinfo.value, (ConnectionError, TimeoutError, OSError))

    def test_an_ephemeral_run_still_fails_loudly_mid_turn(self):
        # Same death, ephemeral thread: nothing survives the subprocess, so a
        # retry is a clean restart and the error must stay transient.
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "exit_mid_turn"))

        async def run():
            return await client.run(
                prompt="hi", model=None, cwd=None, dynamic_tools=[], on_tool_call=_call_tool
            )

        with pytest.raises(ConnectionError):
            asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))


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


def test_server_exit_mid_turn_fails_instead_of_hanging():
    """The turn waiter must be resolved too, not just in-flight requests.

    ``turn/start`` is acknowledged immediately, so by the time the App Server
    dies there is usually nothing left in ``pending`` — only the completion
    future. Leaving it unresolved is invisible under the default
    ``request_timeout`` (it just looks like a slow turn) and hangs forever
    with ``request_timeout=None`` or ``stream_idle_timeout=None``.
    """

    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "exit_mid_turn"))
        return await client.run(
            prompt="quote AMZN",
            model=None,
            cwd=None,
            dynamic_tools=[],
            on_tool_call=_call_tool,
        )

    with pytest.raises(ConnectionError, match="exited before completing"):
        asyncio.run(asyncio.wait_for(run(), timeout=5.0))


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


class TestNativeReview:
    """``review/start``: Codex' own review harness, driven by a typed target.

    Measured live against codex-cli 0.148.0 on a repo with one planted defect:
    all three targets work inline and produce severity-tagged findings with
    file:line; `detached` completes on a different thread and raises an
    approval request the parent never sees, so only inline is wired.
    """

    def test_a_review_target_replaces_the_prompted_turn(self):
        async def run():
            client = CodexAppServerClient(command=(sys.executable, FIXTURE, "native_review"))
            return await client.run(
                prompt="THIS PROMPT IS NOT SENT",
                model=None,
                cwd="C:/work/project",
                dynamic_tools=[],
                on_tool_call=_call_tool,
                review_target={"type": "baseBranch", "branch": "main"},
            )

        result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

        # The fixture asserts the request shape (review/start, inline, target);
        # here we assert the caller gets the findings back like any other turn.
        assert "[P1]" in result.text
        assert (result.input_tokens, result.output_tokens) == (70, 9)


class TestTurnAttributionRaces:
    """The windows around ``turn/start``'s acknowledgement.

    Every one of these was found by Codex reviewing this engine's own diff:
    the turn id arrives in a response, but notifications about that turn — and
    replays of older ones — do not wait for it.
    """

    @staticmethod
    async def _run(scenario: str):
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, scenario))
        return await client.run(
            prompt="carry on",
            model=None,
            cwd="C:/work/project",
            dynamic_tools=[
                {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
            ],
            on_tool_call=_call_tool,
            thread_id="thread-1",
        )

    def test_a_replay_arriving_before_the_ack_is_not_the_answer(self):
        # Accepting it would return the previous turn's text as this call's
        # result — silently, with no error anywhere.
        result = asyncio.run(asyncio.wait_for(self._run("resume_replay_before_ack"), timeout=_TIMEOUT))

        assert result.text == "resumed answer"

    def test_our_own_usage_arriving_before_the_ack_still_counts(self):
        # 100/20 is history; 140/24 and 155/27 are this turn, the first of
        # them reported before we knew our turn id.
        result = asyncio.run(asyncio.wait_for(self._run("resume_usage_before_ack"), timeout=_TIMEOUT))

        assert (result.input_tokens, result.output_tokens) == (55, 7)

    def test_dying_before_the_ack_is_uncertain_not_retryable(self):
        # The server may have accepted the turn already; replaying it could
        # duplicate a committed turn and its tool side effects.
        with pytest.raises(CodexTurnUncertain) as excinfo:
            asyncio.run(asyncio.wait_for(self._run("resume_dies_before_ack"), timeout=_TIMEOUT))

        assert not isinstance(excinfo.value, (ConnectionError, TimeoutError, OSError))
        assert excinfo.value.thread_id == "thread-1"


class TestPreAckCompletion:
    """A completion can outrun its own acknowledgement."""

    def test_our_completion_arriving_before_the_ack_is_still_the_answer(self):
        # Held, not dropped: only the turn id can say whether an early
        # completion is ours or a replay, so it waits for the id instead of
        # being discarded (which hung the call until timeout).
        async def run():
            client = CodexAppServerClient(command=(sys.executable, FIXTURE, "resume_completed_before_ack"))
            return await client.run(
                prompt="carry on",
                model=None,
                cwd="C:/work/project",
                dynamic_tools=[
                    {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
                ],
                on_tool_call=_call_tool,
                thread_id="thread-1",
            )

        result = asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

        assert result.text == "resumed answer"
        assert (result.input_tokens, result.output_tokens) == (55, 7)


def test_a_server_rejection_is_an_error_not_an_uncertain_turn():
    """A JSON-RPC error answer is definitive: nothing was accepted.

    Dressing it as ``CodexTurnUncertain`` would hide the server's actual
    message (invalid review target, unknown thread) behind "resume and
    inspect", and mark a turn that never started as unretryable.
    """

    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "rejects_the_turn"))
        return await client.run(
            prompt="carry on",
            model=None,
            cwd="C:/work/project",
            dynamic_tools=[
                {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
            ],
            on_tool_call=_call_tool,
            thread_id="thread-1",
        )

    with pytest.raises(CodexRequestRejected, match="invalid review target"):
        asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))


def test_a_rejected_turn_is_recorded_as_not_having_run():
    """``sent`` and ``ran`` are different facts, and the difference is recorded
    where a cancellation cannot lose it — in the reader, as the rejection is
    read, rather than where it is caught."""
    progress: dict = {}

    async def run():
        client = CodexAppServerClient(command=(sys.executable, FIXTURE, "rejects_the_turn"))
        return await client.run(
            prompt="carry on",
            model=None,
            cwd="C:/work/project",
            dynamic_tools=[
                {"type": "function", "name": "get_quote", "description": "d", "inputSchema": _QuoteTool._Def.parameters}
            ],
            on_tool_call=_call_tool,
            thread_id="thread-1",
            progress=progress,
        )

    with pytest.raises(CodexRequestRejected):
        asyncio.run(asyncio.wait_for(run(), timeout=_TIMEOUT))

    assert progress["thread_id"] == "thread-1"
    assert progress["turn_sent"] is True
    assert progress["rejected"] is True
