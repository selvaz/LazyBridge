"""Tests for MemoryPanel."""

from __future__ import annotations

import pytest

from lazybridge.gui.memory import MemoryPanel
from lazybridge.memory import Memory


def _populate(mem: Memory) -> None:
    mem._messages.append({"role": "user", "content": "hello there"})
    mem._messages.append({"role": "assistant", "content": "general kenobi"})
    mem._messages.append({"role": "user", "content": "who are you"})
    mem._messages.append({"role": "assistant", "content": "I am a helpful assistant."})


def test_memory_panel_render_state_basic():
    mem = Memory()
    _populate(mem)
    state = MemoryPanel(mem).render_state()
    assert state["message_count"] == 4
    assert state["turn_count"] == 2
    assert state["strategy"] == "auto"
    assert state["summary"] is None
    roles = [m["role"] for m in state["history"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert state["token_estimate"] > 0
    assert state["max_context_tokens"] > 0


def test_memory_panel_label_changes_with_compression():
    mem = Memory()
    _populate(mem)
    panel = MemoryPanel(mem)
    assert "raw" in panel.label
    mem._compressed = "compressed summary"
    assert "compressed" in panel.label


def test_memory_panel_clear_action():
    mem = Memory()
    _populate(mem)
    panel = MemoryPanel(mem)
    panel.handle_action("clear", {})
    assert len(mem) == 0
    assert mem.summary is None


def test_memory_panel_export_history_returns_copy():
    mem = Memory()
    _populate(mem)
    out = MemoryPanel(mem).handle_action("export_history", {})
    assert isinstance(out["history"], list)
    assert len(out["history"]) == 4
    assert out["history"][0]["role"] == "user"


def test_memory_panel_force_compress_runs_without_error():
    mem = Memory(strategy="rolling", max_context_tokens=1, window_turns=1)
    _populate(mem)
    panel = MemoryPanel(mem)
    out = panel.handle_action("force_compress", {})
    assert out["ok"] is True
    # Rolling strategy with a tiny budget should produce a summary.
    assert mem.summary is not None


def test_memory_panel_force_compress_missing_hook():
    class _FakeMem:
        def __init__(self):
            self._messages = []
            self._strategy = "auto"
            self._max_context_tokens = 1000
            self._window_size = 10
            self._compressor = None

        @property
        def history(self):
            return []

        @property
        def summary(self):
            return None

        def __len__(self):
            return 0

        def clear(self):
            pass

    panel = MemoryPanel(_FakeMem())
    with pytest.raises(ValueError):
        panel.handle_action("force_compress", {})


def test_memory_gui_registers_panel():
    from lazybridge.gui import close_server, get_server, install_gui_methods
    from lazybridge.gui._global import _reset_for_tests

    _reset_for_tests()
    install_gui_methods()
    try:
        mem = Memory()
        url = mem.gui(open_browser=False)
        assert "#panel=memory-" in url
        panel_id = url.split("#panel=")[1]
        assert isinstance(get_server().get(panel_id), MemoryPanel)
    finally:
        close_server()
