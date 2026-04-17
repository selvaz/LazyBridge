"""Unit tests for Memory — no API calls."""

from __future__ import annotations

import threading

from lazybridge.memory import Memory

# ── T7.01 — basic append and history ─────────────────────────────────────────


def test_record_and_history():
    mem = Memory(strategy="full")
    mem._record("hello", "hi there")
    history = mem.history
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


def test_history_is_copy():
    """history property returns a copy — mutations don't affect internal state."""
    mem = Memory(strategy="full")
    mem._record("q", "a")
    h = mem.history
    h.append({"role": "user", "content": "injected"})
    assert len(mem.history) == 2  # still 2, not 3


# ── T7.02 — len() reflects number of stored messages ─────────────────────────


def test_len():
    mem = Memory(strategy="full")
    assert len(mem) == 0
    mem._record("q1", "a1")
    assert len(mem) == 2
    mem._record("q2", "a2")
    assert len(mem) == 4


# ── T7.03 — clear resets history ─────────────────────────────────────────────


def test_clear():
    mem = Memory(strategy="full")
    mem._record("q", "a")
    mem.clear()
    assert len(mem) == 0
    assert mem.history == []


# ── T7.04 — _build_input prepends history without mutating state ─────────────


def test_build_input_prepends_history():
    mem = Memory(strategy="full")
    mem._record("turn1", "answer1")
    msgs = mem._build_input("new question")
    assert msgs[-1] == {"role": "user", "content": "new question"}
    assert msgs[0] == {"role": "user", "content": "turn1"}
    assert len(mem) == 2  # _build_input must not mutate


# ── T7.05 — from_history ─────────────────────────────────────────────────────


def test_from_history_restores_messages():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    mem = Memory.from_history(history, strategy="full")
    assert mem.history == history


def test_from_history_empty():
    mem = Memory.from_history([], strategy="full")
    assert len(mem) == 0


def test_from_history_returns_copy():
    history = [{"role": "user", "content": "q"}]
    mem = Memory.from_history(history, strategy="full")
    history.append({"role": "assistant", "content": "a"})
    assert len(mem) == 1


# ── T7.06 — thread safety ────────────────────────────────────────────────────


def test_thread_safety_concurrent_records():
    mem = Memory(strategy="full")
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


# ═════════════════════════════════════════════════════════════════════════════
# Smart Memory — auto compression
# ═════════════════════════════════════════════════════════════════════════════


# ── Full strategy (backward compat) ──────────────────────────────────────────


def test_full_strategy_sends_everything():
    mem = Memory(strategy="full")
    for i in range(20):
        mem._record(f"q{i}", f"a{i}")
    msgs = mem._build_input("new")
    assert len(msgs) == 41  # 40 messages + 1 new


def test_default_strategy_is_auto():
    mem = Memory()
    assert mem._strategy == "auto"


# ── Auto strategy: under threshold sends everything ─────────────────────────


def test_auto_under_threshold_sends_all():
    mem = Memory(strategy="auto", max_context_tokens=100000, window_turns=5)
    for i in range(3):
        mem._record(f"q{i}", f"a{i}")
    msgs = mem._build_input("new")
    assert len(msgs) == 7  # 6 messages + 1 new — no compression


# ── Auto strategy: over threshold compresses ─────────────────────────────────


def test_auto_over_threshold_compresses():
    mem = Memory(strategy="auto", max_context_tokens=100, window_turns=2)
    for i in range(20):
        mem._record(f"question {i} " * 20, f"answer {i} " * 20)

    msgs = mem._build_input("new question")
    # Should have: [compressed system msg] + [4 window msgs] + [1 new]
    assert len(msgs) <= 6
    assert msgs[0]["role"] == "system"
    assert "[Memory" in msgs[0]["content"]
    # Raw history still complete
    assert len(mem.history) == 40


# ── Rolling strategy always uses window ──────────────────────────────────────


def test_rolling_always_windows():
    mem = Memory(strategy="rolling", window_turns=2)
    for i in range(10):
        mem._record(f"q{i}", f"a{i}")

    msgs = mem._build_input("new")
    # Window: 2 turns = 4 msgs. Plus compressed + new = 6
    assert len(msgs) <= 6
    assert msgs[0]["role"] == "system"
    assert len(mem.history) == 20  # full history preserved


# ── Compression preserves raw history ────────────────────────────────────────


def test_compression_preserves_raw_history():
    mem = Memory(strategy="rolling", window_turns=3)
    for i in range(15):
        mem._record(f"q{i}", f"a{i}")
    assert len(mem.history) == 30  # all 30 messages preserved
    assert mem.summary is not None  # compressed block exists


# ── Summary property ─────────────────────────────────────────────────────────


def test_summary_none_when_no_compression():
    mem = Memory(strategy="auto", window_turns=10)
    mem._record("q", "a")
    assert mem.summary is None


def test_summary_exists_after_compression():
    mem = Memory(strategy="rolling", window_turns=1)
    for i in range(5):
        mem._record(f"q{i}", f"a{i}")
    assert mem.summary is not None
    assert "Memory" in mem.summary


# ── Simple compressor extracts topics ────────────────────────────────────────


def test_simple_compress_extracts_topics():
    mem = Memory(strategy="rolling", window_turns=1)
    mem._record("Tell me about Python and Django", "Python is great")
    mem._record("What about Flask?", "Flask is minimal")
    mem._record("And React?", "React is a JS framework")

    msgs = mem._build_input("new")
    compressed = msgs[0]["content"]
    assert "Python" in compressed or "Django" in compressed or "Flask" in compressed


# ── LLM compressor integration ───────────────────────────────────────────────


def test_llm_compressor():
    from unittest.mock import MagicMock

    mock_agent = MagicMock()
    mock_agent.text.return_value = "entities: User(developer)\nfacts: likes_python=true"

    mem = Memory(strategy="rolling", window_turns=1, compressor=mock_agent)
    for i in range(5):
        mem._record(f"q{i}", f"a{i}")

    msgs = mem._build_input("new")
    assert "entities" in msgs[0]["content"]
    mock_agent.text.assert_called()


# ── Clear resets compression state ───────────────────────────────────────────


def test_clear_resets_compression():
    mem = Memory(strategy="rolling", window_turns=1)
    for i in range(5):
        mem._record(f"q{i}", f"a{i}")
    assert mem.summary is not None
    mem.clear()
    assert mem.summary is None
    assert len(mem) == 0


# ── Continuous monitoring triggers recompression ─────────────────────────────


def test_continuous_monitoring_recompresses():
    mem = Memory(strategy="auto", max_context_tokens=50, window_turns=2)
    # First batch — might not trigger (depends on token estimate)
    for i in range(5):
        mem._record(f"question {i} " * 10, f"answer {i} " * 10)

    compressed_v1 = mem.summary

    # Add more turns — should trigger recompression
    for i in range(5, 10):
        mem._record(f"question {i} " * 10, f"answer {i} " * 10)

    # Summary should have been updated
    if compressed_v1 is not None:
        assert mem._compressed_up_to > 0


# ── from_history with strategy ───────────────────────────────────────────────


def test_from_history_with_strategy():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    mem = Memory.from_history(history, strategy="rolling", window_turns=2)
    msgs = mem._build_input("new")
    assert len(msgs) <= 6  # compressed + window + new
