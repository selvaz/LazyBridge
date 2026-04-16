"""Unit tests for Memory — no API calls."""

from __future__ import annotations

import threading

from lazybridge.memory import Memory

# ── T7.01 — basic append and history ─────────────────────────────────────────


def test_record_and_history():
    mem = Memory()
    mem._record("hello", "hi there")
    history = mem.history
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


def test_history_is_copy():
    """history property returns a copy — mutations don't affect internal state."""
    mem = Memory()
    mem._record("q", "a")
    h = mem.history
    h.append({"role": "user", "content": "injected"})
    assert len(mem.history) == 2  # still 2, not 3


# ── T7.02 — len() reflects number of stored messages ─────────────────────────


def test_len():
    mem = Memory()
    assert len(mem) == 0
    mem._record("q1", "a1")
    assert len(mem) == 2
    mem._record("q2", "a2")
    assert len(mem) == 4


# ── T7.03 — clear resets history ─────────────────────────────────────────────


def test_clear():
    mem = Memory()
    mem._record("q", "a")
    mem.clear()
    assert len(mem) == 0
    assert mem.history == []


# ── T7.04 — _build_input prepends history without mutating state ─────────────


def test_build_input_prepends_history():
    mem = Memory()
    mem._record("turn1", "answer1")
    msgs = mem._build_input("new question")
    assert msgs[-1] == {"role": "user", "content": "new question"}
    assert msgs[0] == {"role": "user", "content": "turn1"}
    assert len(mem) == 2  # _build_input must not mutate


# ── T7.05 — from_history: public API to restore serialized memory ─────────────
#
# Audit finding: the module docstring tells users to write mem._messages
# directly (a private attribute).  from_history() is the safe public API.


def test_from_history_restores_messages():
    """Memory.from_history() must exist and restore history correctly."""
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    mem = Memory.from_history(history)
    assert mem.history == history


def test_from_history_empty():
    mem = Memory.from_history([])
    assert len(mem) == 0


def test_from_history_returns_copy():
    """Modifying the original list after construction must not affect the Memory."""
    history = [{"role": "user", "content": "q"}]
    mem = Memory.from_history(history)
    history.append({"role": "assistant", "content": "a"})
    assert len(mem) == 1


# ── T7.06 — thread safety ────────────────────────────────────────────────────


def test_thread_safety_concurrent_records():
    mem = Memory()
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            mem._record(f"q{i}", f"a{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(mem) == 40  # 20 turns × 2 messages each
