"""Tests for v1.0 Memory — no API calls."""

from __future__ import annotations

import threading

from lazybridge.memory import Memory
from lazybridge.core.types import Role


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
    for i in range(15):
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
    for i in range(15):
        m.add("unique_keyword_xyz", "response_abc", tokens=100)
    assert m._summary != ""

def test_messages_includes_summary_prefix():
    m = Memory(strategy="sliding", max_tokens=1)
    for i in range(15):
        m.add("q", "a", tokens=100)
    msgs = m.messages()
    full_text = " ".join(
        msg.content if isinstance(msg.content, str) else ""
        for msg in msgs
    )
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
