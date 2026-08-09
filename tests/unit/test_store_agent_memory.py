"""Unit tests for Store.write_memory / read_memory / delete_memory.

Dedicated agent_memory table (composite (agent_id, session_key) identity),
separate from the generic key/value ``store`` table. Backs
``Memory(store=..., key=...)``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from lazybridge.store import Store

# ── in-memory ────────────────────────────────────────────────────────────


def test_write_then_read_memory_roundtrips():
    store = Store()
    store.write_memory("agent-a", turns=[{"user": "hi", "assistant": "hello", "token_estimate": 2}], summary="")
    loaded = store.read_memory("agent-a")
    assert loaded is not None
    assert loaded["turns"] == [{"user": "hi", "assistant": "hello", "token_estimate": 2}]
    assert loaded["summary"] == ""
    assert isinstance(loaded["updated_at"], float)


def test_read_memory_missing_returns_none():
    store = Store()
    assert store.read_memory("nonexistent") is None


def test_write_memory_overwrites_on_conflict():
    store = Store()
    store.write_memory("agent-a", turns=[{"user": "q1", "assistant": "a1", "token_estimate": 1}])
    store.write_memory("agent-a", turns=[{"user": "q2", "assistant": "a2", "token_estimate": 1}], summary="prior ctx")
    loaded = store.read_memory("agent-a")
    assert loaded["turns"] == [{"user": "q2", "assistant": "a2", "token_estimate": 1}]
    assert loaded["summary"] == "prior ctx"


def test_session_key_isolates_conversations_under_same_agent_id():
    store = Store()
    store.write_memory("agent-a", turns=[{"user": "u1", "assistant": "a1", "token_estimate": 1}], session_key="user-1")
    store.write_memory("agent-a", turns=[{"user": "u2", "assistant": "a2", "token_estimate": 1}], session_key="user-2")
    assert store.read_memory("agent-a", session_key="user-1")["turns"][0]["user"] == "u1"
    assert store.read_memory("agent-a", session_key="user-2")["turns"][0]["user"] == "u2"


def test_delete_memory_removes_entry():
    store = Store()
    store.write_memory("agent-a", turns=[{"user": "q", "assistant": "a", "token_estimate": 1}])
    store.delete_memory("agent-a")
    assert store.read_memory("agent-a") is None


def test_delete_memory_missing_is_noop():
    store = Store()
    store.delete_memory("nonexistent")  # must not raise


def test_read_memory_returns_copy_not_shared_reference():
    store = Store()
    turns = [{"user": "q", "assistant": "a", "token_estimate": 1}]
    store.write_memory("agent-a", turns=turns)
    loaded = store.read_memory("agent-a")
    loaded["turns"].append({"user": "mutated", "assistant": "x", "token_estimate": 1})
    assert len(store.read_memory("agent-a")["turns"]) == 1


# ── SQLite-backed ────────────────────────────────────────────────────────


def test_sqlite_write_then_read_memory_roundtrips():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "agents.sqlite")
        store = Store(db=db)
        store.write_memory("agent-a", turns=[{"user": "hi", "assistant": "hello", "token_estimate": 2}])
        loaded = store.read_memory("agent-a")
        assert loaded["turns"] == [{"user": "hi", "assistant": "hello", "token_estimate": 2}]
        store.close()


def test_sqlite_memory_survives_reopen():
    """Simulates a process restart: new Store instance, same db file."""
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "agents.sqlite")
        store1 = Store(db=db)
        store1.write_memory("agent-a", turns=[{"user": "order 4821", "assistant": "noted", "token_estimate": 2}])
        store1.close()

        store2 = Store(db=db)
        loaded = store2.read_memory("agent-a")
        assert loaded is not None
        assert loaded["turns"][0]["user"] == "order 4821"
        store2.close()


def test_sqlite_write_memory_overwrites_on_conflict():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "agents.sqlite")
        store = Store(db=db)
        store.write_memory("agent-a", turns=[{"user": "q1", "assistant": "a1", "token_estimate": 1}])
        store.write_memory("agent-a", turns=[{"user": "q2", "assistant": "a2", "token_estimate": 1}])
        loaded = store.read_memory("agent-a")
        assert loaded["turns"] == [{"user": "q2", "assistant": "a2", "token_estimate": 1}]
        store.close()


def test_sqlite_session_key_isolates_conversations():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "agents.sqlite")
        store = Store(db=db)
        store.write_memory("agent-a", turns=[{"user": "u1", "assistant": "a1", "token_estimate": 1}], session_key="s1")
        store.write_memory("agent-a", turns=[{"user": "u2", "assistant": "a2", "token_estimate": 1}], session_key="s2")
        assert store.read_memory("agent-a", session_key="s1")["turns"][0]["user"] == "u1"
        assert store.read_memory("agent-a", session_key="s2")["turns"][0]["user"] == "u2"
        store.close()


def test_sqlite_delete_memory_removes_entry():
    with tempfile.TemporaryDirectory() as d:
        db = str(Path(d) / "agents.sqlite")
        store = Store(db=db)
        store.write_memory("agent-a", turns=[{"user": "q", "assistant": "a", "token_estimate": 1}])
        store.delete_memory("agent-a")
        assert store.read_memory("agent-a") is None
        store.close()


def test_agent_memory_does_not_leak_into_generic_store_table():
    """write_memory must not create a row visible via the generic store.* API."""
    store = Store()
    store.write_memory("agent-a", turns=[{"user": "q", "assistant": "a", "token_estimate": 1}])
    assert store.keys() == []
    assert store.read_all() == {}
