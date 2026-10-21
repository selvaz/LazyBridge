from __future__ import annotations

import webbrowser

from lazybridge.ext import _browser


def test_launch_is_suppressed_under_pytest(monkeypatch):
    calls = []
    monkeypatch.setattr(webbrowser, "open", lambda url, **kw: calls.append(url) or True)
    assert _browser.browser_launch_allowed() is False
    assert _browser.open_browser("http://127.0.0.1:1/") is False
    assert calls == []


def test_env_switch_suppresses_and_absence_allows(monkeypatch):
    calls = []
    monkeypatch.setattr(webbrowser, "open", lambda url, **kw: calls.append(url) or True)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("LAZYBRIDGE_NO_BROWSER", "1")
    assert _browser.open_browser("http://127.0.0.1:1/") is False
    monkeypatch.delenv("LAZYBRIDGE_NO_BROWSER")
    assert _browser.open_browser("http://127.0.0.1:2/") is True
    assert calls == ["http://127.0.0.1:2/"]


def test_launch_failure_never_raises(monkeypatch):
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    def boom(url, **kw):
        raise RuntimeError("no display")

    monkeypatch.setattr(webbrowser, "open", boom)
    assert _browser.open_browser("http://127.0.0.1:3/") is False
