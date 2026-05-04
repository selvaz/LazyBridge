"""Server-level smoke tests: routing, auth, static serving, SSE handshake."""

from __future__ import annotations

import http.client
import threading
from urllib.parse import urlparse

import pytest

from lazybridge.ext.viz.exporter import EventHub
from lazybridge.ext.viz.server import VizServer


@pytest.fixture
def server():
    hub = EventHub()
    s = VizServer(
        hub,
        graph_provider=lambda: {"nodes": [{"id": "a", "name": "a", "type": "agent"}], "edges": []},
        store_provider=lambda: {"k": "v"},
        meta_provider=lambda: {"mode": "live", "session_id": "abcd1234"},
        port=0,
    )
    s.start()
    yield s, hub
    s.stop()


def _client(s):
    return http.client.HTTPConnection(s.host, s.port, timeout=2.0)


def test_index_served_without_token(server):
    s, _ = server
    c = _client(s)
    c.request("GET", "/")
    r = c.getresponse()
    body = r.read()
    assert r.status == 200
    assert r.getheader("Content-Type", "").startswith("text/html")
    assert b"LazyBridge" in body


def test_static_assets_served(server):
    s, _ = server
    c = _client(s)
    c.request("GET", "/static/styles.css")
    r = c.getresponse()
    assert r.status == 200
    assert r.getheader("Content-Type", "").startswith("text/css")


def test_static_path_traversal_blocked(server):
    s, _ = server
    c = _client(s)
    c.request("GET", "/static/../server.py")
    r = c.getresponse()
    r.read()
    assert r.status == 404


def test_api_requires_token(server):
    s, _ = server
    c = _client(s)
    c.request("GET", "/api/graph")
    r = c.getresponse()
    r.read()
    assert r.status == 401


def test_api_with_token_returns_payload(server):
    s, _ = server
    c = _client(s)
    c.request("GET", f"/api/graph?t={s.token}")
    r = c.getresponse()
    body = r.read()
    assert r.status == 200
    assert b'"nodes"' in body


def test_meta_and_store_endpoints(server):
    s, _ = server
    for path in ("/api/meta", "/api/store"):
        c = _client(s)
        c.request("GET", f"{path}?t={s.token}")
        r = c.getresponse()
        r.read()
        assert r.status == 200, path


def test_sse_handshake_returns_event_stream_content_type(server):
    s, hub = server

    def _open():
        c = _client(s)
        c.request("GET", f"/api/events?t={s.token}")
        r = c.getresponse()
        # First read should not block forever — heartbeat fires within a few ms
        assert r.status == 200
        assert r.getheader("Content-Type", "").startswith("text/event-stream")
        # Ensure we close out
        c.close()

    t = threading.Thread(target=_open)
    t.start()
    t.join(timeout=3.0)
    assert not t.is_alive()


def test_url_includes_host_port_and_token(server):
    s, _ = server
    parsed = urlparse(s.url)
    assert parsed.scheme == "http"
    assert parsed.hostname == "127.0.0.1"
    assert s.token in s.url
