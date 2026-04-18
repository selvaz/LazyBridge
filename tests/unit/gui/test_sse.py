"""Tests for the SSE notification path on GuiServer."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from lazybridge.gui._panel import Panel
from lazybridge.gui._server import GuiServer


class _StatePanel(Panel):
    kind = "agent"

    def __init__(self, panel_id: str = "p") -> None:
        self._id = panel_id
        self._counter = 0

    @property
    def id(self) -> str:
        return self._id

    def render_state(self) -> dict:
        return {
            "counter": self._counter,
            "name": self._id,
            "provider": "fake",
            "model": "m",
            "system": "",
            "tools": [],
            "available_tools": [],
        }

    def handle_action(self, action, args):
        if action == "bump":
            self._counter += 1
            self.notify()
            return {"counter": self._counter}
        return super().handle_action(action, args)


@pytest.fixture
def server():
    s = GuiServer(open_browser=False)
    try:
        yield s
    finally:
        s.close()


def _open_sse(server: GuiServer, timeout: float = 2.0):
    url = f"http://127.0.0.1:{server.port}/api/events?t={server.token}"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    return urllib.request.urlopen(req, timeout=timeout)


def _read_event(stream, timeout: float = 1.0) -> dict | None:
    """Read one ``event: refresh\\ndata: <json>\\n\\n`` block, ignoring keep-alives."""
    deadline = time.monotonic() + timeout
    block = b""
    while time.monotonic() < deadline:
        line = stream.readline()
        if not line:
            return None
        if line == b"\n":
            if block:
                return _parse_block(block)
            block = b""
            continue
        block += line
    return None


def _parse_block(block: bytes) -> dict | None:
    event_name = None
    data = None
    for raw in block.splitlines():
        if raw.startswith(b":"):
            continue  # comment / keep-alive
        if raw.startswith(b"event:"):
            event_name = raw[len(b"event:") :].strip().decode()
        elif raw.startswith(b"data:"):
            data = raw[len(b"data:") :].strip().decode()
    if event_name != "refresh" or data is None:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def test_sse_endpoint_requires_no_unusual_headers_and_stays_open(server):
    stream = _open_sse(server)
    try:
        # First payload is the ``: connected`` comment — confirm we can
        # read at least one byte.
        first = stream.readline()
        assert first.startswith(b":")
    finally:
        stream.close()


def test_register_unregister_emits_list_event(server):
    stream = _open_sse(server)
    try:
        # Drain the initial ``: connected`` comment + blank line.
        stream.readline()
        stream.readline()
        server.register(_StatePanel("a"))
        evt = _read_event(stream, timeout=2.0)
        assert evt == {"type": "list"}
        server.unregister("a")
        evt = _read_event(stream, timeout=2.0)
        assert evt == {"type": "list"}
    finally:
        stream.close()


def test_panel_notify_emits_state_event_with_panel_id(server):
    p = _StatePanel("watch-me")
    server.register(p)
    stream = _open_sse(server)
    try:
        stream.readline()

        stream.readline()
        # Trigger a state change via the panel's own action.
        p.handle_action("bump", {})
        evt = _read_event(stream, timeout=2.0)
        assert evt == {"type": "state", "panel_id": "watch-me"}
    finally:
        stream.close()


def test_multiple_subscribers_each_receive(server):
    p = _StatePanel("multi")
    server.register(p)
    stream_a = _open_sse(server)
    stream_b = _open_sse(server)
    try:
        for s in (stream_a, stream_b):
            s.readline()
            s.readline()
        p.handle_action("bump", {})
        evt_a = _read_event(stream_a, timeout=2.0)
        evt_b = _read_event(stream_b, timeout=2.0)
        assert evt_a == {"type": "state", "panel_id": "multi"}
        assert evt_b == {"type": "state", "panel_id": "multi"}
    finally:
        stream_a.close()
        stream_b.close()


def test_close_terminates_subscribers(server):
    stream = _open_sse(server)
    stream.readline()

    stream.readline()

    received: list[bytes] = []

    def _drain():
        try:
            for line in stream:
                received.append(line)
        except Exception:
            pass

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    server.close()
    t.join(timeout=2.0)
    # Reader thread must have exited (server closed cleanly).
    assert not t.is_alive()


def test_keepalive_emitted_periodically_does_not_break_reader(server):
    """Stream stays open while idle (we don't wait the full 15 s — just confirm
    that the initial connect comment arrived and the reader didn't crash)."""
    stream = _open_sse(server)
    try:
        first = stream.readline()
        assert first.startswith(b":")
        blank = stream.readline()
        assert blank == b"\n"
    finally:
        stream.close()


def test_unregister_clears_panel_notifier(server):
    p = _StatePanel("x")
    server.register(p)
    assert p._notifier is not None
    server.unregister("x")
    assert p._notifier is None


def test_pipeline_panel_notifies_on_completion():
    """End-to-end: pipeline run pushes refresh events as it progresses."""
    from types import SimpleNamespace

    from lazybridge.gui.pipeline import PipelinePanel
    from lazybridge.lazy_session import LazySession

    sess = LazySession()
    agent = SimpleNamespace(
        id="a",
        name="alpha",
        session=sess,
        _provider_name="anthropic",
        _model_name="m",
    )

    def _run(args):
        sess.events.log("agent_start", agent_id="a", agent_name="alpha")
        sess.events.log(
            "agent_finish",
            agent_id="a",
            agent_name="alpha",
            method="chat",
            stop_reason="end_turn",
            n_steps=1,
        )
        return "ok"

    tool = SimpleNamespace(
        name="t",
        description="x",
        _is_pipeline_tool=True,
        _pipeline=SimpleNamespace(
            mode="chain",
            participants=(agent,),
            combiner=None,
            concurrency_limit=None,
            step_timeout=None,
            guidance=None,
        ),
        run=_run,
    )

    server = GuiServer(open_browser=False)
    try:
        panel = PipelinePanel(tool)
        server.register(panel)
        stream = _open_sse(server)
        try:
            stream.readline()

            stream.readline()
            panel.handle_action("run", {"task": "go"})
            # Collect a handful of events.
            events: list[dict] = []
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and len(events) < 3:
                evt = _read_event(stream, timeout=1.0)
                if evt is None:
                    continue
                events.append(evt)
            # We expect at least one state notification with our panel_id.
            assert any(e.get("panel_id") == panel.id for e in events), events
        finally:
            stream.close()
    finally:
        server.close()
