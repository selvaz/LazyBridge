"""Unit tests for LazyStore — in-memory backend, no API calls."""
from __future__ import annotations

import threading
import pytest

from lazybridge.lazy_store import LazyStore


# ── T2.01 — write + read round-trip ──────────────────────────────────────────

def test_write_read_roundtrip():
    store = LazyStore()
    store.write("key", "value")
    assert store.read("key") == "value"


def test_write_read_various_types():
    store = LazyStore()
    store.write("int", 42)
    store.write("list", [1, 2, 3])
    store.write("dict", {"a": 1})
    store.write("none", None)
    assert store.read("int") == 42
    assert store.read("list") == [1, 2, 3]
    assert store.read("dict") == {"a": 1}
    assert store.read("none") is None


def test_read_missing_returns_default():
    store = LazyStore()
    assert store.read("missing") is None
    assert store.read("missing", default="fallback") == "fallback"


# ── T2.02 — __setitem__ delegates to write (same JSON validation) ─────────────

def test_setitem_delegates_to_write():
    store = LazyStore()
    store["key"] = "via setitem"
    assert store.read("key") == "via setitem"


def test_setitem_non_serializable_raises():
    store = LazyStore()
    with pytest.raises(TypeError, match="not JSON-serializable"):
        store["bad"] = object()          # non-serializable


# ── T2.03 — __getitem__: missing key raises KeyError ─────────────────────────

def test_getitem_missing_raises_keyerror():
    store = LazyStore()
    with pytest.raises(KeyError):
        _ = store["nonexistent"]


def test_getitem_existing_key():
    store = LazyStore()
    store.write("x", 99)
    assert store["x"] == 99


# ── T2.04 — write: non-JSON-serializable value raises TypeError ───────────────

def test_write_non_serializable_raises():
    store = LazyStore()
    with pytest.raises(TypeError, match="not JSON-serializable"):
        store.write("key", lambda: None)


# ── T2.05 — write with agent_id → read_by_agent returns correct subset ────────

def test_write_agent_id_filtering():
    store = LazyStore()
    store.write("a1", "val1", agent_id="agent_a")
    store.write("a2", "val2", agent_id="agent_a")
    store.write("b1", "val3", agent_id="agent_b")
    result = store.read_by_agent("agent_a")
    assert set(result.keys()) == {"a1", "a2"}
    assert store.read_by_agent("agent_b") == {"b1": "val3"}


# ── T2.06 — delete removes key; no error on deleting non-existent key ─────────

def test_delete_removes_key():
    store = LazyStore()
    store.write("key", "val")
    store.delete("key")
    assert store.read("key") is None
    assert "key" not in store


def test_delete_nonexistent_no_error():
    store = LazyStore()
    store.delete("ghost")   # should not raise


# ── T2.07 — clear empties all keys ───────────────────────────────────────────

def test_clear_empties_store():
    store = LazyStore()
    store.write("a", 1)
    store.write("b", 2)
    store.clear()
    assert store.read_all() == {}


# ── T2.08 — __contains__ ─────────────────────────────────────────────────────

def test_contains_true_and_false():
    store = LazyStore()
    store.write("present", True)
    assert "present" in store
    assert "absent" not in store


# ── T2.09 — to_text format ───────────────────────────────────────────────────

def test_to_text_format():
    store = LazyStore()
    store.write("city", "Rome")
    store.write("temp", 22)
    text = store.to_text()
    assert text.startswith("[shared store]")
    assert "city" in text
    assert "Rome" in text
    assert "temp" in text


def test_to_text_empty_returns_empty_string():
    store = LazyStore()
    assert store.to_text() == ""


def test_to_text_filtered_keys():
    store = LazyStore()
    store.write("a", 1)
    store.write("b", 2)
    text = store.to_text(keys=["a"])
    assert "a" in text
    assert "b" not in text


# ── T2.10 — thread safety: 50 threads write distinct keys ────────────────────

def test_thread_safety_concurrent_writes():
    store = LazyStore()
    n = 50
    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            store.write(f"key_{i}", i)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    result = store.read_all()
    assert len(result) == n
    for i in range(n):
        assert result[f"key_{i}"] == i


# ── T2.11 — async methods: awrite / aread / aread_all / akeys ────────────────

import asyncio
import pytest


@pytest.mark.asyncio
async def test_async_write_read():
    store = LazyStore()
    await store.awrite("x", 99)
    assert await store.aread("x") == 99


@pytest.mark.asyncio
async def test_async_read_default():
    store = LazyStore()
    assert await store.aread("missing", "default") == "default"


@pytest.mark.asyncio
async def test_async_read_all():
    store = LazyStore()
    await store.awrite("a", 1)
    await store.awrite("b", 2)
    result = await store.aread_all()
    assert result == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_async_keys():
    store = LazyStore()
    await store.awrite("k1", "v1")
    await store.awrite("k2", "v2")
    keys = await store.akeys()
    assert sorted(keys) == ["k1", "k2"]
