"""Tests for HumanInputPanel + panel_input_fn on the shared server."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from lazybridge.gui import close_server, get_server, install_gui_methods
from lazybridge.gui._global import _reset_for_tests
from lazybridge.gui.human_panel import HumanInputPanel, panel_input_fn


@pytest.fixture(autouse=True)
def _fresh_server():
    _reset_for_tests()
    install_gui_methods()
    yield
    close_server()


def _base_and_token():
    server = get_server(open_browser=False)
    return f"http://127.0.0.1:{server.port}/", server.token


def test_panel_input_fn_registers_on_shared_server():
    fn = panel_input_fn(name="reviewer", open_browser=False)
    assert callable(fn)
    assert isinstance(fn.panel, HumanInputPanel)
    server = get_server(open_browser=False)
    assert any(p.id == "human-reviewer" for p in server.panels())
    fn.panel.close()


def test_ask_round_trip_over_http():
    fn = panel_input_fn(name="r", open_browser=False)
    panel = fn.panel
    base, token = _base_and_token()
    panel_id = panel.id

    result: list[str] = []

    def _ask():
        result.append(panel.ask("approve?"))

    t = threading.Thread(target=_ask)
    t.start()

    # Poll until the prompt is visible.
    deadline = time.monotonic() + 2.0
    seq = None
    while time.monotonic() < deadline:
        body = urllib.request.urlopen(f"{base}api/panel/{panel_id}?t={token}", timeout=2).read()
        data = json.loads(body)
        if data.get("prompt") is not None:
            seq = data["seq"]
            break
        time.sleep(0.02)
    assert seq is not None

    # Submit via the shared server.
    body = json.dumps({"action": "submit", "args": {"seq": seq, "response": "yes"}}).encode()
    req = urllib.request.Request(
        f"{base}api/panel/{panel_id}/action?t={token}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=2).read())
    assert payload == {"accepted": True}

    t.join(timeout=2)
    assert result == ["yes"]
    panel.close()


def test_panel_label_reflects_waiting_state():
    fn = panel_input_fn(name="x", open_browser=False)
    panel = fn.panel
    assert panel.label == "x"

    # Start a background ask and verify the label flips to "waiting".
    def _ask():
        try:
            panel.ask("hello")
        except RuntimeError:
            pass

    t = threading.Thread(target=_ask)
    t.start()
    time.sleep(0.05)
    assert panel.label == "x · waiting"
    panel.close()
    t.join(timeout=2)


def test_submit_with_stale_seq_rejected():
    fn = panel_input_fn(name="s", open_browser=False)
    panel = fn.panel
    base, token = _base_and_token()
    body = json.dumps({"action": "submit", "args": {"seq": 9999, "response": "x"}}).encode()
    req = urllib.request.Request(
        f"{base}api/panel/{panel.id}/action?t={token}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=2).read())
    assert payload == {"accepted": False}
    panel.close()


def test_close_unblocks_ask():
    fn = panel_input_fn(name="c", open_browser=False)
    panel = fn.panel
    errors: list[Exception] = []

    def _ask():
        try:
            panel.ask("wait")
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=_ask)
    t.start()
    time.sleep(0.05)
    panel.close()
    t.join(timeout=2)
    assert errors and isinstance(errors[0], RuntimeError)


def test_ask_after_close_raises():
    fn = panel_input_fn(name="d", open_browser=False)
    panel = fn.panel
    panel.close()
    with pytest.raises(RuntimeError):
        panel.ask("nope")


def test_supervisor_agent_integration():
    from lazybridge import SupervisorAgent

    fn = panel_input_fn(name="sup", open_browser=False)
    panel = fn.panel
    supervisor = SupervisorAgent(name="sup", input_fn=fn)

    base, token = _base_and_token()

    result: list[object] = []

    def _run():
        result.append(supervisor.chat("research output"))

    t = threading.Thread(target=_run)
    t.start()
    deadline = time.monotonic() + 2.0
    seq = None
    while time.monotonic() < deadline:
        body = urllib.request.urlopen(f"{base}api/panel/{panel.id}?t={token}", timeout=2).read()
        data = json.loads(body)
        if data.get("prompt") is not None:
            seq = data["seq"]
            break
        time.sleep(0.02)
    assert seq is not None
    body = json.dumps({"action": "submit", "args": {"seq": seq, "response": "continue"}}).encode()
    req = urllib.request.Request(
        f"{base}api/panel/{panel.id}/action?t={token}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=2)
    t.join(timeout=2)
    assert result and result[0].content == "research output"
    panel.close()
