"""Regression tests for Wave 2 of the deep audit (H1, H2, H3, H5)."""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.types import CompletionResponse, Message, Role, StreamChunk, UsageStats
from lazybridge.gui.agent import AgentPanel
from lazybridge.lazy_agent import LazyAgent


class AnthropicProvider:
    def get_default_max_tokens(self) -> int:  # pragma: no cover
        return 4096


def _bare_agent(name="a"):
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider = AnthropicProvider()
    mock_exec.model = "m"
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = name
    agent.description = None
    agent.system = ""
    agent.context = None
    agent.tools = []
    agent.native_tools = []
    agent.output_schema = None
    agent._last_output = None
    agent._last_response = None
    agent.session = None
    agent._log = None
    return agent


# ---------------------------------------------------------------------------
# H1 — stream _last_output reflects bytes yielded, even on early break
# ---------------------------------------------------------------------------


def test_stream_and_track_updates_last_output_incrementally():
    agent = _bare_agent()
    agent._last_output = "STALE"

    def _gen():
        yield StreamChunk(delta="hel")
        yield StreamChunk(delta="lo ")
        yield StreamChunk(delta="world")

    it = agent._stream_and_track(_gen())
    # Consume one chunk at a time and assert _last_output grows.
    chunk1 = next(it)
    assert chunk1.delta == "hel"
    assert agent._last_output == "hel"
    chunk2 = next(it)
    assert chunk2.delta == "lo "
    assert agent._last_output == "hello "
    # Close the iterator early (simulates a `break` in the caller).
    it.close()
    # Already-yielded bytes must be preserved, NOT reverted to stale value.
    assert agent._last_output == "hello "


async def test_astream_and_track_updates_last_output_incrementally():
    agent = _bare_agent()
    agent._last_output = "STALE"

    async def _agen():
        yield StreamChunk(delta="foo")
        yield StreamChunk(delta="bar")

    it = agent._astream_and_track(_agen())
    c1 = await it.__anext__()
    assert c1.delta == "foo"
    assert agent._last_output == "foo"
    await it.__anext__()  # consume second chunk
    assert agent._last_output == "foobar"
    await it.aclose()
    assert agent._last_output == "foobar"


# ---------------------------------------------------------------------------
# H2 — OpenAI sync + async streams emit a final chunk even if finish_reason
# is absent.  We drive the logic by stubbing self._client.chat.completions.create
# to yield chunks without a finish_reason.
# ---------------------------------------------------------------------------


def test_openai_stream_emits_incomplete_final_chunk_when_interrupted():
    from lazybridge.core.providers.openai import OpenAIProvider
    from lazybridge.core.types import CompletionRequest

    # Build a minimal provider instance without calling __init__.
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._api_key = "test-key"
    provider._default_model = "gpt-test"
    provider._beta_overrides = {}

    # Stub client.chat.completions.create to yield two chunks and stop
    # without finish_reason.
    class _FakeChoice:
        def __init__(self, delta):
            self.delta = delta
            self.finish_reason = None

    class _FakeDelta:
        def __init__(self, content=None):
            self.content = content
            self.tool_calls = None

    class _FakeChunk:
        def __init__(self, content=None):
            self.usage = None
            self.choices = [_FakeChoice(_FakeDelta(content))] if content else []
            self.model = "gpt-test"

    def _fake_stream():
        yield _FakeChunk("par")
        yield _FakeChunk("tial")
        # Stream ends without a finish_reason.

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_stream()
    provider._client = mock_client
    # Force the Chat Completions path — we only patched H2 there.
    provider._use_responses_api = lambda request: False  # type: ignore[method-assign]

    # Use a minimal request — structured_output=None avoids validation path.
    request = CompletionRequest(
        model="gpt-test",
        messages=[Message(role=Role.USER, content="hi")],
        stream=True,
    )

    # Drive the sync-stream method.
    chunks = list(provider.stream(request))
    final = chunks[-1]
    assert final.is_final is True, f"expected a final chunk when finish_reason is missing; got chunks={chunks}"
    assert final.stop_reason == "incomplete"


# ---------------------------------------------------------------------------
# H3 — Google tool-call IDs don't collide when the same function is called
# twice in one turn.
# ---------------------------------------------------------------------------


def test_google_synthesize_fc_id_is_unique_across_repeated_calls():
    """Unit test for the H3 fix helper, independent of the Gemini SDK."""
    from lazybridge.core.providers.google import _synthesize_fc_id

    class _FC:
        def __init__(self, name, with_id=None):
            self.name = name
            if with_id is not None:
                self.id = with_id

    # Two calls to the same function WITHOUT an id — must produce distinct ids.
    a = _synthesize_fc_id(_FC("search"), 0)
    b = _synthesize_fc_id(_FC("search"), 1)
    assert a != b
    assert a.startswith("search-") and b.startswith("search-")

    # A call WITH an id preserves that id regardless of index.
    c = _synthesize_fc_id(_FC("search", with_id="abc123"), 5)
    assert c == "abc123"


# ---------------------------------------------------------------------------
# H5 — AgentPanel.test action no longer blocks the HTTP handler thread
# ---------------------------------------------------------------------------


def test_agent_panel_test_action_is_non_blocking_by_default():
    agent = _bare_agent()
    slow_event = {"released": False}

    def _slow_chat(msg):
        while not slow_event["released"]:
            time.sleep(0.005)
        return CompletionResponse(content="done", usage=UsageStats())

    agent.chat = _slow_chat
    panel = AgentPanel(agent)
    out = panel.handle_action("test", {"mode": "chat", "message": "hi"})
    # Returned quickly with an acknowledgment, NOT the final result.
    assert out["started"] is True
    # Status is "running".
    state = panel.render_state()
    assert state["last_test"]["status"] == "running"
    # Label carries the "· running" marker.
    assert panel.label.endswith("· running")

    # Release the chat and wait for completion.
    slow_event["released"] = True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = panel.render_state()
        if state["last_test"]["status"] == "done":
            break
        time.sleep(0.01)
    assert state["last_test"]["status"] == "done"
    assert state["last_test"]["content"] == "done"


def test_agent_panel_test_action_sync_path_still_returns_result_inline():
    agent = _bare_agent()
    resp = CompletionResponse(content="inline", usage=UsageStats(input_tokens=1, output_tokens=2))
    agent.chat = MagicMock(return_value=resp)
    panel = AgentPanel(agent)
    out = panel.handle_action("test", {"mode": "chat", "message": "hi", "sync": True})
    assert out["content"] == "inline"
    # No background run state was stored in sync mode.
    assert "last_test" not in panel.render_state()


def test_agent_panel_rejects_second_test_while_first_running():
    agent = _bare_agent()
    event_flag = {"done": False}

    def _hold(msg):
        while not event_flag["done"]:
            time.sleep(0.005)
        return CompletionResponse(content="x", usage=UsageStats())

    agent.chat = _hold
    panel = AgentPanel(agent)
    panel.handle_action("test", {"mode": "chat", "message": "one"})
    try:
        with pytest.raises(ValueError):
            panel.handle_action("test", {"mode": "chat", "message": "two"})
    finally:
        event_flag["done"] = True


def test_agent_panel_clear_test_drops_last_test():
    agent = _bare_agent()
    agent.chat = MagicMock(return_value=CompletionResponse(content="hi", usage=UsageStats()))
    panel = AgentPanel(agent)
    panel.handle_action("test", {"mode": "chat", "message": "go"})
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if panel.render_state().get("last_test", {}).get("status") == "done":
            break
        time.sleep(0.01)
    assert "last_test" in panel.render_state()
    panel.handle_action("clear_test", {})
    assert "last_test" not in panel.render_state()


def test_agent_panel_test_error_captured_as_error_state():
    agent = _bare_agent()

    def _boom(msg):
        raise RuntimeError("kaboom")

    agent.chat = _boom
    panel = AgentPanel(agent)
    panel.handle_action("test", {"mode": "chat", "message": "go"})
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if panel.render_state().get("last_test", {}).get("status") == "error":
            break
        time.sleep(0.01)
    assert panel.render_state()["last_test"]["status"] == "error"
    assert "kaboom" in panel.render_state()["last_test"]["error"]
