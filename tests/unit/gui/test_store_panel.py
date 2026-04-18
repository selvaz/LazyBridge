"""Tests for StorePanel."""

from __future__ import annotations

import pytest

from lazybridge.gui.store import StorePanel
from lazybridge.lazy_store import LazyStore


def _store():
    s = LazyStore()
    s.write("hello", "world", agent_id="a1")
    s.write("n", 42, agent_id="a1")
    s.write("lst", [1, 2, 3], agent_id="a2")
    return s


def test_store_panel_render_state_lists_entries():
    panel = StorePanel(_store())
    state = panel.render_state()
    assert state["key_count"] == 3
    keys = {e["key"] for e in state["entries"]}
    assert keys == {"hello", "n", "lst"}
    agents = {e["agent_id"] for e in state["entries"]}
    assert agents == {"a1", "a2"}


def test_store_panel_read():
    panel = StorePanel(_store())
    out = panel.handle_action("read", {"key": "hello"})
    assert out == {"key": "hello", "value": "world"}


def test_store_panel_read_missing_key_returns_none_value():
    panel = StorePanel(_store())
    out = panel.handle_action("read", {"key": "ghost"})
    assert out == {"key": "ghost", "value": None}


def test_store_panel_write_plain():
    store = LazyStore()
    panel = StorePanel(store)
    out = panel.handle_action("write", {"key": "greeting", "value": "hi"})
    assert out["ok"] is True
    assert store.read("greeting") == "hi"


def test_store_panel_write_as_json():
    store = LazyStore()
    panel = StorePanel(store)
    panel.handle_action("write", {"key": "config", "value": '{"n":1,"k":["a"]}', "as_json": True})
    assert store.read("config") == {"n": 1, "k": ["a"]}


def test_store_panel_write_invalid_json_raises():
    store = LazyStore()
    panel = StorePanel(store)
    with pytest.raises(ValueError):
        panel.handle_action("write", {"key": "x", "value": "{bad", "as_json": True})


def test_store_panel_delete_removes_key():
    store = _store()
    panel = StorePanel(store)
    out = panel.handle_action("delete", {"key": "hello"})
    assert out == {"ok": True, "key": "hello"}
    assert "hello" not in store


def test_store_panel_delete_missing_is_soft_failure():
    panel = StorePanel(_store())
    out = panel.handle_action("delete", {"key": "missing"})
    assert out == {"ok": False, "reason": "not found"}


def test_store_panel_read_all_returns_snapshot():
    panel = StorePanel(_store())
    out = panel.handle_action("read_all", {})
    assert out["all"]["hello"] == "world"
    assert out["all"]["n"] == 42
    assert out["all"]["lst"] == [1, 2, 3]


def test_store_panel_read_requires_key():
    panel = StorePanel(_store())
    with pytest.raises(ValueError):
        panel.handle_action("read", {})


def test_store_gui_registers_panel():
    from lazybridge.gui import close_server, get_server, install_gui_methods
    from lazybridge.gui._global import _reset_for_tests

    _reset_for_tests()
    install_gui_methods()
    try:
        store = _store()
        url = store.gui(open_browser=False)
        assert "#panel=store-" in url
        server = get_server()
        panel_id = url.split("#panel=")[1]
        assert server.get(panel_id) is not None
    finally:
        close_server()
