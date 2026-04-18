"""Regression tests for audit L6 — GUI JS served from _static/app.js."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from lazybridge.gui._server import GuiServer, _STATIC_JS_PATH, _load_static_js


@pytest.fixture
def server():
    srv = GuiServer(open_browser=False)
    try:
        yield srv
    finally:
        srv.close()


def test_static_js_file_is_present_on_disk():
    """The file must ship in the source tree — wheel packaging depends
    on it being alongside the module (audit L6)."""
    assert _STATIC_JS_PATH.exists(), f"{_STATIC_JS_PATH} missing"
    content = _STATIC_JS_PATH.read_text()
    # Sanity: essential landmarks from the extracted client.
    assert "PANEL_RENDERERS" in content
    assert "window.LB_TOKEN" in content


def test_page_includes_bootstrap_and_external_script(server):
    page = urllib.request.urlopen(f"http://127.0.0.1:{server.port}/").read().decode()
    assert "window.LB_TOKEN =" in page
    assert 'src="/static/app.js?t=' in page
    # The deprecated inline-client block must be type="text/plain" so
    # the browser never executes it.
    assert 'data-disabled="original-inline-client"' in page
    assert 'type="text/plain"' in page


def test_static_js_served_with_token(server):
    url = f"http://127.0.0.1:{server.port}/static/app.js?t={server.token}"
    resp = urllib.request.urlopen(url)
    assert resp.status == 200
    assert "javascript" in resp.headers.get("Content-Type", "")
    body = resp.read().decode()
    assert "PANEL_RENDERERS" in body
    assert len(body) > 10_000  # full client, not a stub


def test_static_js_requires_token(server):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"http://127.0.0.1:{server.port}/static/app.js")
    assert e.value.code == 401


def test_load_static_js_caches_one_read():
    a = _load_static_js()
    b = _load_static_js()
    # Identity: the one-time cache returns the same string object.
    assert a is b
