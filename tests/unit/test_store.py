"""Unit tests for Store.items(prefix=) and EncryptedStoreAdapter.items(prefix=)."""

from __future__ import annotations

import importlib.util

import pytest

from lazybridge.store import Store
from lazybridge.store.encryption import EncryptedStoreAdapter


def _populate(store: Store) -> None:
    store.write("pulse:task:abc", {"status": "scheduled"})
    store.write("pulse:task:def", {"status": "running"})
    store.write("pulse:taskx:xyz", {"status": "completed"})  # different prefix — must NOT match "pulse:task:"
    store.write("pulse:event:e1", {"task_id": "abc"})
    store.write("other:key", "value")


# ── in-memory ──────────────────────────────────────────────────────────────


def test_items_all_memory() -> None:
    store = Store()
    _populate(store)
    result = dict(store.items())
    assert set(result) == {"pulse:task:abc", "pulse:task:def", "pulse:taskx:xyz", "pulse:event:e1", "other:key"}


def test_items_prefix_match_memory() -> None:
    store = Store()
    _populate(store)
    result = dict(store.items(prefix="pulse:task:"))
    assert set(result) == {"pulse:task:abc", "pulse:task:def"}
    # prefix "pulse:task:" must not match "pulse:taskx:"
    assert "pulse:taskx:xyz" not in result


def test_items_prefix_no_match_memory() -> None:
    store = Store()
    _populate(store)
    assert store.items(prefix="pulse:nonexistent:") == []


def test_items_empty_prefix_returns_all_memory() -> None:
    store = Store()
    _populate(store)
    assert len(store.items(prefix="")) == 5


def test_items_none_prefix_returns_all_memory() -> None:
    store = Store()
    _populate(store)
    assert len(store.items(prefix=None)) == 5


def test_items_returns_copies_memory() -> None:
    store = Store()
    store.write("k", {"mutable": True})
    [(_, v)] = store.items()
    v["mutable"] = False
    # The stored value must not be affected
    assert store.read("k")["mutable"] is True


# ── SQLite ──────────────────────────────────────────────────────────────────


def test_items_all_sqlite(tmp_path) -> None:
    db = str(tmp_path / "s.db")
    with Store(db=db) as store:
        _populate(store)
        result = dict(store.items())
    assert set(result) == {"pulse:task:abc", "pulse:task:def", "pulse:taskx:xyz", "pulse:event:e1", "other:key"}


def test_items_prefix_match_sqlite(tmp_path) -> None:
    db = str(tmp_path / "s.db")
    with Store(db=db) as store:
        _populate(store)
        result = dict(store.items(prefix="pulse:task:"))
    assert set(result) == {"pulse:task:abc", "pulse:task:def"}
    assert "pulse:taskx:xyz" not in result


def test_items_prefix_no_match_sqlite(tmp_path) -> None:
    db = str(tmp_path / "s.db")
    with Store(db=db) as store:
        _populate(store)
        assert store.items(prefix="pulse:nonexistent:") == []


def test_items_empty_prefix_sqlite(tmp_path) -> None:
    db = str(tmp_path / "s.db")
    with Store(db=db) as store:
        _populate(store)
        assert len(store.items(prefix="")) == 5


# ── parity: in-memory <-> SQLite ──────────────────────────────────────────


def test_items_prefix_parity(tmp_path) -> None:
    db = str(tmp_path / "s.db")
    mem = Store()
    sql = Store(db=db)
    for store in (mem, sql):
        _populate(store)
    mem_result = sorted(mem.items(prefix="pulse:task:"))
    sql_result = sorted(sql.items(prefix="pulse:task:"))
    assert mem_result == sql_result
    mem.close()
    sql.close()


# ── EncryptedStoreAdapter ────────────────────────────────────────────────────


_HAS_CRYPTO = importlib.util.find_spec("cryptography") is not None


@pytest.mark.skipif(not _HAS_CRYPTO, reason="cryptography not installed")
def test_encrypted_adapter_items_prefix(tmp_path) -> None:
    from cryptography.fernet import Fernet

    key = Fernet.generate_key()
    db = str(tmp_path / "enc.db")
    inner = Store(db=db)
    adapter = EncryptedStoreAdapter(inner, key=key)
    adapter.write("pulse:task:a", {"x": 1})
    adapter.write("pulse:task:b", {"x": 2})
    adapter.write("other:key", "value")

    result = dict(adapter.items(prefix="pulse:task:"))
    assert set(result) == {"pulse:task:a", "pulse:task:b"}
    assert result["pulse:task:a"] == {"x": 1}
    inner.close()
