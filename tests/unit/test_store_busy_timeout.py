"""Store lock wait and slow-write logging (a 5 s wait failed a shared-Store job after minutes of work)."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time

import pytest

from lazybridge import Store
from lazybridge import store as store_module


def test_default_busy_timeout_is_thirty_seconds(tmp_path, monkeypatch):
    monkeypatch.delenv("LAZYBRIDGE_STORE_BUSY_TIMEOUT", raising=False)
    store = Store(str(tmp_path / "s.sqlite"))
    try:
        assert store._conn().execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    finally:
        store.close()


@pytest.mark.parametrize(("raw", "expected"), [("2.5", 2500), ("0", 30000), ("bad", 30000)])
def test_busy_timeout_follows_the_environment(tmp_path, monkeypatch, raw, expected):
    monkeypatch.setenv("LAZYBRIDGE_STORE_BUSY_TIMEOUT", raw)
    store = Store(str(tmp_path / "s.sqlite"))
    try:
        assert store._conn().execute("PRAGMA busy_timeout").fetchone()[0] == expected
    finally:
        store.close()


def test_a_write_waits_for_a_held_lock_and_logs_the_wait(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LAZYBRIDGE_STORE_SLOW_WRITE", "0.2")
    path = tmp_path / "s.sqlite"
    store = Store(str(path))
    store.write("warm", 1)
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    released = threading.Event()

    def release() -> None:
        time.sleep(0.6)
        holder.execute("ROLLBACK")
        released.set()

    threading.Thread(target=release, daemon=True).start()
    try:
        with caplog.at_level(logging.WARNING, logger="lazybridge.store"):
            store.write("key", "value")
        assert released.is_set()
        assert store.read("key") == "value"
        assert any("slow Store write on 'key'" in record.getMessage() for record in caplog.records)
    finally:
        holder.close()
        store.close()


def test_fast_writes_are_not_logged(tmp_path, caplog):
    store = Store(str(tmp_path / "s.sqlite"))
    try:
        with caplog.at_level(logging.WARNING, logger="lazybridge.store"):
            store.write("a", 1)
            assert store.compare_and_swap("a", 1, 2)
        assert not [r for r in caplog.records if r.name == "lazybridge.store"]
    finally:
        store.close()


def test_module_exports_the_defaults():
    assert store_module.DEFAULT_BUSY_TIMEOUT_SECONDS == 30.0
    assert store_module.DEFAULT_SLOW_WRITE_SECONDS == 1.0
