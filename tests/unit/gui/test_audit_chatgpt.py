"""Regression tests for the ChatGPT-audit findings (F3, F4, F5)."""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from lazybridge.core.types import CompletionResponse, ToolCall, UsageStats
from lazybridge.lazy_agent import LazyAgent
from lazybridge.lazy_session import LazySession
from lazybridge.lazy_store import LazyStore
from lazybridge.pipeline_builders import _clear_checkpoint


class AnthropicProvider:
    def get_default_max_tokens(self, model=None) -> int:  # pragma: no cover
        return 4096


def _bare_agent():
    with patch("lazybridge.core.executor.Executor.__init__", return_value=None):
        agent = LazyAgent.__new__(LazyAgent)
    mock_exec = MagicMock()
    mock_exec.provider = AnthropicProvider()
    mock_exec.model = "m"
    agent._executor = mock_exec
    agent.id = str(uuid.uuid4())
    agent.name = "a"
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
# F4 — _clear_checkpoint deletes instead of writing None
# ---------------------------------------------------------------------------


def test_clear_checkpoint_removes_key_from_store():
    store = LazyStore()
    store.write("_ckpt:demo", {"step": 2, "output": "partial"})
    assert "_ckpt:demo" in store
    _clear_checkpoint(store, "_ckpt:demo")
    assert "_ckpt:demo" not in store, (
        "checkpoint key should be DELETED, not left as a None-tombstone"
    )


def test_clear_checkpoint_is_idempotent_when_key_missing():
    store = LazyStore()
    # No raise, no complaint when the key is already absent.
    _clear_checkpoint(store, "_ckpt:missing")
    assert "_ckpt:missing" not in store


def test_clear_checkpoint_tolerates_store_without_delete():
    """If a user-supplied store lacks .delete(), fall back to write(None)."""
    class _WriteOnlyStore:
        def __init__(self) -> None:
            self.writes: list[tuple[str, object]] = []

        def write(self, key, value):
            self.writes.append((key, value))

    s = _WriteOnlyStore()
    _clear_checkpoint(s, "_ckpt:legacy")
    assert s.writes == [("_ckpt:legacy", None)]


# ---------------------------------------------------------------------------
# F5 — per-tool timeout in loop / aloop
# ---------------------------------------------------------------------------


def test_execute_tool_honours_tool_timeout_sync():
    agent = _bare_agent()
    # LazyTool-like stub whose run() sleeps longer than the timeout.
    slow = MagicMock()
    slow.name = "slow"
    slow.run = lambda args, parent=None: time.sleep(1.0) or "done"
    registry = {"slow": slow}
    call = ToolCall(id="1", name="slow", arguments={})
    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as e:
        agent._execute_tool(call, registry, runner=None, tool_timeout=0.05)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"timeout didn't fire early, elapsed={elapsed}"
    assert "tool 'slow' exceeded tool_timeout" in str(e.value)


def test_execute_tool_passes_through_without_timeout():
    agent = _bare_agent()
    fast = MagicMock()
    fast.name = "fast"
    fast.run = lambda args, parent=None: "ok"
    registry = {"fast": fast}
    call = ToolCall(id="1", name="fast", arguments={})
    assert agent._execute_tool(call, registry, runner=None) == "ok"
    assert agent._execute_tool(call, registry, runner=None, tool_timeout=None) == "ok"
    assert agent._execute_tool(call, registry, runner=None, tool_timeout=0) == "ok"


async def test_aexecute_tool_honours_tool_timeout_async():
    import asyncio

    agent = _bare_agent()
    slow = MagicMock()
    slow.name = "slow"

    async def _slow_arun(args, parent=None):
        await asyncio.sleep(1.0)
        return "done"

    slow.arun = _slow_arun
    registry = {"slow": slow}
    call = ToolCall(id="1", name="slow", arguments={})
    with pytest.raises(TimeoutError):
        await agent._aexecute_tool(call, registry, runner=None, tool_timeout=0.05)


def test_loop_plumbs_tool_timeout_to_executor():
    """loop(..., tool_timeout=...) must reach _execute_tool."""
    from lazybridge.lazy_tool import LazyTool

    agent = _bare_agent()

    # Stub chat() so the generator emits ONE tool call then terminates.
    first = CompletionResponse(
        content="",
        tool_calls=[ToolCall(id="1", name="slow", arguments={})],
        stop_reason="tool_use",
        usage=UsageStats(),
    )
    final = CompletionResponse(content="all done", usage=UsageStats())
    agent.chat = MagicMock(side_effect=[first, final])

    def slow(q: str = "") -> str:
        """Slow tool."""
        time.sleep(1.0)
        return "late"

    tool = LazyTool.from_function(slow)

    captured: dict[str, object] = {}
    original = agent._execute_tool

    def _spy(*a, **kw):
        captured["tool_timeout"] = kw.get("tool_timeout")
        return original(*a, **kw)

    agent._execute_tool = _spy  # type: ignore[method-assign]

    # Call loop with a tight timeout; the tool will exceed it and the
    # loop will raise TimeoutError.
    with pytest.raises(TimeoutError):
        agent.loop("go", tools=[tool], tool_timeout=0.05)
    assert captured.get("tool_timeout") == 0.05


# ---------------------------------------------------------------------------
# F3 — EventLog redactor hook
# ---------------------------------------------------------------------------


def test_eventlog_redactor_rewrites_tool_arguments():
    captured: list[dict] = []

    def _redact(event_type: str, data: dict) -> dict:
        out = dict(data)
        if "arguments" in out and isinstance(out["arguments"], dict):
            out["arguments"] = {k: "[REDACTED]" for k in out["arguments"]}
        return out

    sess = LazySession(redact=_redact)
    # The redactor runs before the exporter.
    from lazybridge.exporters import CallbackExporter

    sess.events.add_exporter(CallbackExporter(captured.append))
    sess.events.log(
        "tool_call",
        agent_id="a",
        agent_name="n",
        name="search",
        arguments={"api_key": "sk-secret", "query": "mail"},
    )
    assert len(captured) == 1
    data = captured[0]["data"]
    assert data["arguments"] == {"api_key": "[REDACTED]", "query": "[REDACTED]"}


def test_eventlog_redactor_exception_keeps_original_payload():
    """A crashing redactor must NOT drop events — log warning + use original."""
    def _broken(event_type, data):
        raise RuntimeError("oops")

    captured: list[dict] = []
    sess = LazySession(redact=_broken)
    from lazybridge.exporters import CallbackExporter

    sess.events.add_exporter(CallbackExporter(captured.append))
    sess.events.log(
        "tool_call", agent_id="a", agent_name="n",
        name="search", arguments={"q": "x"},
    )
    assert len(captured) == 1
    assert captured[0]["data"]["arguments"] == {"q": "x"}


def test_eventlog_redactor_default_is_noop():
    captured: list[dict] = []
    sess = LazySession()  # no redact=
    from lazybridge.exporters import CallbackExporter

    sess.events.add_exporter(CallbackExporter(captured.append))
    sess.events.log(
        "tool_call", agent_id="a", agent_name="n",
        name="search", arguments={"q": "x"},
    )
    assert captured[0]["data"]["arguments"] == {"q": "x"}
