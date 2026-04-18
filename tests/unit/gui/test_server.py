"""Tests for the shared GuiServer + Panel plumbing (no LazyBridge classes)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from lazybridge.gui._panel import Panel
from lazybridge.gui._server import GuiServer


class _EchoPanel(Panel):
    kind = "agent"

    def __init__(self, panel_id: str = "demo") -> None:
        self._id = panel_id
        self.actions: list[tuple[str, dict]] = []

    @property
    def id(self) -> str:
        return self._id

    def render_state(self) -> dict:
        return {"name": "demo", "provider": "fake", "model": "m", "system": "", "tools": [], "available_tools": []}

    def handle_action(self, action, args):
        self.actions.append((action, args))
        if action == "boom":
            raise ValueError("nope")
        return {"action": action, "args": args}


@pytest.fixture
def server():
    s = GuiServer(open_browser=False)
    try:
        yield s
    finally:
        s.close()


def _get(url):
    return urllib.request.urlopen(url, timeout=2).read()


def _post(url, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=2).read()


def test_page_served_with_title(server):
    body = _get(f"http://127.0.0.1:{server.port}/").decode()
    assert "LazyBridge GUI" in body
    assert json.dumps(server.token) in body


def test_empty_panels(server):
    data = json.loads(_get(f"http://127.0.0.1:{server.port}/api/panels?t={server.token}"))
    assert data == {"panels": []}


def test_register_and_list(server):
    panel = _EchoPanel("p1")
    url = server.register(panel)
    assert "#panel=p1" in url
    data = json.loads(_get(f"http://127.0.0.1:{server.port}/api/panels?t={server.token}"))
    assert data == {"panels": [{"id": "p1", "kind": "agent", "label": "p1", "group": "Agents"}]}


def test_panel_state_route(server):
    server.register(_EchoPanel("abc"))
    data = json.loads(_get(f"http://127.0.0.1:{server.port}/api/panel/abc?t={server.token}"))
    assert data["id"] == "abc"
    assert data["kind"] == "agent"
    assert data["name"] == "demo"


def test_panel_unknown_404(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"http://127.0.0.1:{server.port}/api/panel/ghost?t={server.token}")
    assert e.value.code == 404


def test_action_success(server):
    p = _EchoPanel("x")
    server.register(p)
    body = _post(
        f"http://127.0.0.1:{server.port}/api/panel/x/action?t={server.token}", {"action": "touch", "args": {"k": 1}}
    )
    assert json.loads(body) == {"action": "touch", "args": {"k": 1}}
    assert p.actions == [("touch", {"k": 1})]


def test_action_validation_error_is_400(server):
    server.register(_EchoPanel("x"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"http://127.0.0.1:{server.port}/api/panel/x/action?t={server.token}", {"action": "boom", "args": {}})
    assert e.value.code == 400


def test_token_enforced_on_api_and_action(server):
    server.register(_EchoPanel("y"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"http://127.0.0.1:{server.port}/api/panels")
    assert e.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(f"http://127.0.0.1:{server.port}/api/panel/y/action", {"action": "touch", "args": {}})
    assert e.value.code == 401


def test_unregister(server):
    server.register(_EchoPanel("gone"))
    server.unregister("gone")
    data = json.loads(_get(f"http://127.0.0.1:{server.port}/api/panels?t={server.token}"))
    assert data == {"panels": []}


def test_healthz_public(server):
    server.register(_EchoPanel("p"))
    body = json.loads(_get(f"http://127.0.0.1:{server.port}/healthz"))
    assert body == {"ok": True, "panels": 1, "closed": False}
