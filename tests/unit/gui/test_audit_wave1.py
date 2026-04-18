"""Regression tests for Wave 1 of the deep audit (C1, H4, L3, L4)."""

from __future__ import annotations

import logging
import threading
import time

import pytest

from lazybridge.gui._panel import Panel
from lazybridge.gui._server import GuiServer
from lazybridge.gui.memory import MemoryPanel
from lazybridge.gui.store import GUI_AGENT_ID, StorePanel
from lazybridge.lazy_store import LazyStore
from lazybridge.memory import Memory


# ---------------------------------------------------------------------------
# C1 — token must not appear in the INFO log
# ---------------------------------------------------------------------------


def test_startup_log_does_not_contain_token(caplog):
    with caplog.at_level(logging.INFO, logger="lazybridge.gui._server"):
        srv = GuiServer(open_browser=False)
        try:
            assert any("listening at" in rec.getMessage() for rec in caplog.records)
            joined = " ".join(rec.getMessage() for rec in caplog.records)
            # Token must not appear in any INFO record.
            assert srv.token not in joined
            # The tokenised URL must stay accessible through the property.
            assert srv.token in srv.url
        finally:
            srv.close()


# ---------------------------------------------------------------------------
# H4 — MemoryPanel.force_compress acquires Memory._lock
# ---------------------------------------------------------------------------


def test_force_compress_holds_memory_lock():
    """MemoryPanel.force_compress must hold Memory._lock while running.

    We instrument the memory's `_maybe_recompress` to sleep briefly while
    holding `self._lock`, then try to acquire the lock from another thread
    and verify it's contested — i.e. the panel is holding the lock.
    """
    mem = Memory(strategy="rolling", max_context_tokens=1, window_turns=1)
    for _ in range(6):
        mem._messages.append({"role": "user", "content": "hi " * 100})
        mem._messages.append({"role": "assistant", "content": "ok " * 100})

    original = mem._maybe_recompress
    entered = threading.Event()

    def _slow():
        entered.set()
        time.sleep(0.05)
        original()

    mem._maybe_recompress = _slow  # type: ignore[method-assign]
    panel = MemoryPanel(mem)

    acquired_while_running = {"value": True}  # assume True; flip to False if we fail
    t = threading.Thread(target=panel.handle_action, args=("force_compress", {}))
    t.start()
    assert entered.wait(timeout=1.0)
    # Try to acquire the memory lock now — must fail because the panel holds it.
    acquired = mem._lock.acquire(blocking=False)
    acquired_while_running["value"] = acquired
    if acquired:
        mem._lock.release()
    t.join(timeout=2.0)
    assert not t.is_alive()
    assert acquired_while_running["value"] is False, (
        "MemoryPanel.force_compress did not hold memory._lock during recompression"
    )


def test_force_compress_still_produces_summary():
    mem = Memory(strategy="rolling", max_context_tokens=1, window_turns=1)
    for _ in range(6):
        mem._messages.append({"role": "user", "content": "hi"})
        mem._messages.append({"role": "assistant", "content": "ok"})
    out = MemoryPanel(mem).handle_action("force_compress", {})
    assert out["ok"] is True
    assert mem.summary is not None


def test_force_compress_missing_hook_still_raises():
    class _NoHook:
        _messages: list = []
        _strategy = "auto"
        _max_context_tokens = 1000
        _window_size = 10
        _compressor = None

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

    with pytest.raises(ValueError):
        MemoryPanel(_NoHook()).handle_action("force_compress", {})


# ---------------------------------------------------------------------------
# L3 — Panel.notify logs failures instead of swallowing them silently
# ---------------------------------------------------------------------------


class _BoomPanel(Panel):
    kind = "agent"

    @property
    def id(self) -> str:
        return "boom"

    def render_state(self) -> dict:
        return {}


def test_notify_failure_is_logged_not_swallowed(caplog):
    p = _BoomPanel()

    def _broken_notifier(kind, panel_id):
        raise RuntimeError("simulated notifier crash")

    p._notifier = _broken_notifier
    with caplog.at_level(logging.DEBUG, logger="lazybridge.gui._panel"):
        p.notify()  # must not raise
    # At debug level, the exception shows up with a traceback.
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("Panel.notify()" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# L4 — StorePanel uses a non-colliding agent_id
# ---------------------------------------------------------------------------


def test_store_panel_write_uses_reserved_agent_id():
    store = LazyStore()
    StorePanel(store).handle_action("write", {"key": "k", "value": "v"})
    entry = store.read_entry("k")
    assert entry is not None
    assert entry.agent_id == GUI_AGENT_ID
    # And a real agent named "gui" would not collide:
    assert entry.agent_id != "gui"
