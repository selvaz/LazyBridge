"""Tests for v1.0 Memory — no API calls."""

from __future__ import annotations

import threading
import time

import pytest

from lazybridge.core.types import Role
from lazybridge.memory import Memory
from lazybridge.store import Store

# ── basic add / text ──────────────────────────────────────────────────────────


def test_add_and_text():
    m = Memory()
    m.add("hello", "world")
    txt = m.text()
    assert "hello" in txt
    assert "world" in txt


def test_empty_memory_text_is_empty():
    m = Memory()
    assert m.text() == ""


def test_clear_resets():
    m = Memory()
    m.add("q", "a")
    m.clear()
    assert m.text() == ""
    assert m.messages() == []


def test_multiple_turns():
    m = Memory()
    m.add("q1", "a1")
    m.add("q2", "a2")
    txt = m.text()
    assert "q1" in txt
    assert "a2" in txt


# ── messages() returns Message list ──────────────────────────────────────────


def test_messages_alternates_roles():
    m = Memory()
    m.add("user question", "assistant answer")
    msgs = m.messages()
    assert len(msgs) >= 2
    roles = [msg.role for msg in msgs]
    assert Role.USER in roles
    assert Role.ASSISTANT in roles


def test_messages_returns_new_list():
    m = Memory()
    m.add("q", "a")
    msgs1 = m.messages()
    msgs1.append(None)
    assert len(m.messages()) != len(msgs1)


# ── strategy="none" ───────────────────────────────────────────────────────────


def test_strategy_none_keeps_all():
    m = Memory(strategy="none")
    for i in range(25):
        m.add(f"q{i}", f"a{i}", tokens=1000)
    assert len(m._turns) == 25


def test_strategy_none_no_summary():
    m = Memory(strategy="none")
    for i in range(25):
        m.add(f"q{i}", f"a{i}", tokens=1000)
    assert m._summary == ""


# ── strategy="sliding" ───────────────────────────────────────────────────────


def test_strategy_sliding_caps_at_10():
    m = Memory(strategy="sliding", max_tokens=1)
    for i in range(15):
        m.add(f"q{i}", f"a{i}", tokens=100)
    assert len(m._turns) <= 10


def test_strategy_sliding_creates_summary():
    m = Memory(strategy="sliding", max_tokens=1)
    for _ in range(15):
        m.add("question", "answer", tokens=200)
    assert m._summary != ""


# ── strategy="auto" ───────────────────────────────────────────────────────────


def test_auto_under_threshold_no_compression():
    m = Memory(strategy="auto", max_tokens=10000)
    m.add("q", "a", tokens=10)
    assert len(m._turns) == 1
    assert m._summary == ""


def test_auto_over_threshold_compresses():
    m = Memory(strategy="auto", max_tokens=100)
    for i in range(15):
        m.add(f"q{i}", f"a{i}", tokens=50)
    assert len(m._turns) <= 10


def test_auto_default_strategy():
    m = Memory()
    assert m.strategy == "auto"


# ── summary ───────────────────────────────────────────────────────────────────


def test_summary_non_empty_after_compression():
    m = Memory(strategy="sliding", max_tokens=1)
    for _ in range(15):
        m.add("unique_keyword_xyz", "response_abc", tokens=100)
    assert m._summary != ""


def test_messages_includes_summary_prefix():
    m = Memory(strategy="sliding", max_tokens=1)
    for _ in range(15):
        m.add("q", "a", tokens=100)
    msgs = m.messages()
    full_text = " ".join(msg.content if isinstance(msg.content, str) else "" for msg in msgs)
    assert "earlier" in full_text.lower() or len(msgs) > 2


# ── thread safety ─────────────────────────────────────────────────────────────


def test_thread_safety_concurrent_adds():
    m = Memory(strategy="none")
    errors: list[Exception] = []

    def _add():
        try:
            for i in range(50):
                m.add(f"q{i}", f"a{i}", tokens=1)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_add) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(m._turns) == 250


# ── live view ─────────────────────────────────────────────────────────────────


def test_text_reflects_latest_state():
    m = Memory()
    m.add("initial", "response")
    t1 = m.text()
    m.add("new question", "new answer")
    t2 = m.text()
    assert "new question" in t2
    assert len(t2) >= len(t1)


# ── summarizer: sync / async / Agent-like — all bridge correctly ─────────────
#
# ``Memory.add()`` is sync but accepts an ``async def`` summarizer.
# An un-awaited coroutine would stringify to ``"<coroutine object at
# 0x…>"`` with a runtime warning, so the implementation drives the
# awaitable to completion (running-loop-safe via a
# worker thread when needed) and falls back to keyword extraction on
# raise — never silent garbage.


def _force_compression_turns() -> list[tuple[str, str]]:
    """Build a turns list long enough that strategy='summary' compresses
    on the next add()."""
    return [(f"user-{i}", f"assistant-{i}") for i in range(11)]


def test_summarizer_sync_callable_is_used_verbatim():
    calls: list[str] = []

    def sync_sum(prompt: str) -> str:
        calls.append(prompt)
        return "SYNC-SUMMARY"

    m = Memory(strategy="summary", summarizer=sync_sum)
    for u, a in _force_compression_turns():
        m.add(u, a)
    assert len(calls) >= 1
    assert m._summary == "SYNC-SUMMARY"


def test_summarizer_async_callable_is_driven_to_completion():
    """An ``async def`` summarizer's coroutine must be awaited, not
    silently stringified."""
    calls: list[str] = []

    async def async_sum(prompt: str) -> str:
        calls.append(prompt)
        return "ASYNC-SUMMARY"

    m = Memory(strategy="summary", summarizer=async_sum)
    for u, a in _force_compression_turns():
        m.add(u, a)

    assert len(calls) >= 1
    assert m._summary == "ASYNC-SUMMARY"
    # Explicitly guard against regression — the broken path produces
    # something with "coroutine" in it.
    assert "coroutine" not in m._summary.lower()


def test_summarizer_async_callable_raising_falls_back_to_keywords():
    async def async_boom(prompt: str) -> str:
        raise RuntimeError("upstream down")

    m = Memory(strategy="summary", summarizer=async_boom)
    for u, a in _force_compression_turns():
        m.add(u, a)
    # Rule fallback prefix — compression didn't just die.
    assert m._summary.startswith("[Earlier conversation covered:")


def test_summarizer_object_with_text_method_is_stringified():
    """A sync callable returning an object exposing ``.text()`` (e.g.
    an Envelope-like) has ``.text()`` consulted — preserves the shape
    that Agents return via ``__call__``."""

    class _ResultLike:
        def text(self) -> str:
            return "FROM-TEXT"

    def sync_sum(prompt: str) -> _ResultLike:
        return _ResultLike()

    m = Memory(strategy="summary", summarizer=sync_sum)
    for u, a in _force_compression_turns():
        m.add(u, a)
    assert m._summary == "FROM-TEXT"


# ── store= / key= persistence ────────────────────────────────────────────


def test_store_without_key_raises():
    store = Store()
    with pytest.raises(ValueError):
        Memory(store=store)


def test_store_persists_across_instances():
    """New Memory instance, same store + key — simulates a process restart."""
    store = Store()
    m1 = Memory(store=store, key="agent-a")
    m1.add("hi, I'm Marco", "Nice to meet you, Marco.")

    m2 = Memory(store=store, key="agent-a")
    assert "Marco" in m2.text()


def test_store_different_keys_do_not_collide():
    store = Store()
    m1 = Memory(store=store, key="agent-a")
    m1.add("agent a turn", "a-reply")
    m2 = Memory(store=store, key="agent-b")
    m2.add("agent b turn", "b-reply")

    reloaded_a = Memory(store=store, key="agent-a")
    reloaded_b = Memory(store=store, key="agent-b")
    assert "agent a turn" in reloaded_a.text()
    assert "agent b turn" not in reloaded_a.text()
    assert "agent b turn" in reloaded_b.text()


def test_store_session_key_separates_conversations_for_same_key():
    store = Store()
    m1 = Memory(store=store, key="agent-a", session_key="user-1")
    m1.add("user 1 says hi", "hello user 1")
    m2 = Memory(store=store, key="agent-a", session_key="user-2")
    m2.add("user 2 says hi", "hello user 2")

    reloaded_1 = Memory(store=store, key="agent-a", session_key="user-1")
    assert "user 1 says hi" in reloaded_1.text()
    assert "user 2 says hi" not in reloaded_1.text()


def test_store_fresh_key_is_empty():
    """A key that was never written looks like a brand-new Memory — no crash."""
    store = Store()
    m = Memory(store=store, key="never-seen-before")
    assert m.text() == ""


def test_store_clear_persists_the_wipe():
    store = Store()
    m1 = Memory(store=store, key="agent-a")
    m1.add("q", "a")
    m1.clear()

    m2 = Memory(store=store, key="agent-a")
    assert m2.text() == ""


def test_store_amend_last_persists():
    store = Store()
    m1 = Memory(store=store, key="agent-a")
    m1.add("q", "draft answer")
    m1.amend_last("corrected answer")

    m2 = Memory(store=store, key="agent-a")
    assert "corrected answer" in m2.text()
    assert "draft answer" not in m2.text()


class _SlowFirstWriteStore:
    """Wraps a real Store; the FIRST write_memory() call sleeps before
    delegating, every later call is immediate. Used to prove persistence
    stays ordered with mutations: if a later add() (fast write) could ever
    land before an earlier add()'s (slow) write, the earlier write would
    finish last and silently overwrite the newer state — the exact race
    flagged in review on lazybridge/memory.py's Memory._persist_locked.
    """

    def __init__(self, inner: Store) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self._call_count = 0

    def read_memory(self, *args, **kwargs):
        return self._inner.read_memory(*args, **kwargs)

    def write_memory(self, *args, **kwargs):
        with self._lock:
            self._call_count += 1
            is_first_call = self._call_count == 1
        if is_first_call:
            time.sleep(0.05)
        self._inner.write_memory(*args, **kwargs)


def test_persist_stays_ordered_with_mutations_under_concurrency():
    """Regression test: a slow first write must not overwrite a fast second one."""
    inner = Store()
    slow_store = _SlowFirstWriteStore(inner)
    mem = Memory(store=slow_store, key="agent-a")

    def first_add():
        mem.add("first question", "first reply")

    def second_add():
        time.sleep(0.01)  # let first_add() start, and hit the slow write, first
        mem.add("second question", "second reply")

    t1 = threading.Thread(target=first_add)
    t2 = threading.Thread(target=second_add)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    persisted = inner.read_memory("agent-a")
    assert persisted is not None
    users = [t["user"] for t in persisted["turns"]]
    assert "first question" in users
    assert "second question" in users


def test_store_survives_compression():
    """Persisted state after compression carries the summary, not just raw turns."""
    store = Store()
    m1 = Memory(strategy="sliding", store=store, key="agent-a")
    for i in range(15):
        m1.add(f"q{i}", f"a{i}")

    m2 = Memory(strategy="sliding", store=store, key="agent-a")
    # Sliding keeps only the most recent turns in `_turns`, but the reloaded
    # instance must still see the recent tail — not an empty buffer.
    assert "q14" in m2.text()
