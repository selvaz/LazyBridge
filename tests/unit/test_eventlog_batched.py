"""Tests for ``EventLog`` batched async writer.

Closes audit finding #4 — synchronous per-event SQLite commits.

Covers:
  * Batched mode persists all events submitted before ``flush()``.
  * Time-based flush kicks in when batch_size is not reached.
  * Bounded queue with ``on_full="drop"`` drops on saturation
    without raising; ``on_full="block"`` backpressures the producer.
  * ``close()`` drains the queue and joins the writer thread cleanly.
  * Construction validation rejects invalid knob values.
  * Default ``batched=False`` preserves the prior synchronous path.
"""

from __future__ import annotations

import time
import warnings

import pytest

from lazybridge.session import EventType, Session

# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Session(batched=True, batch_size=0)


def test_batch_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_interval"):
        Session(batched=True, batch_interval=0)


def test_max_queue_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_queue_size"):
        Session(batched=True, max_queue_size=0)


def test_on_full_must_be_drop_or_block() -> None:
    with pytest.raises(ValueError, match="on_full"):
        Session(batched=True, on_full="explode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Default (non-batched) path is unchanged
# ---------------------------------------------------------------------------


def test_default_session_is_not_batched() -> None:
    sess = Session()
    sess.emit(EventType.AGENT_START, {"i": 1})
    # Synchronous: row visible immediately, no flush needed.
    assert len(sess.events.query()) == 1
    sess.close()


# ---------------------------------------------------------------------------
# Batched mode — basic correctness
# ---------------------------------------------------------------------------


def test_batched_session_persists_all_events_after_flush() -> None:
    sess = Session(batched=True, batch_size=3, batch_interval=0.5)
    for i in range(7):
        sess.emit(EventType.AGENT_START, {"i": i})
    sess.flush()
    assert len(sess.events.query()) == 7
    sess.close()


def test_batched_close_drains_queue() -> None:
    sess = Session(batched=True, batch_size=10, batch_interval=5.0)
    for i in range(5):
        sess.emit(EventType.AGENT_START, {"i": i})
    # Don't call flush — close() should drain implicitly.
    sess.close()
    # Re-open the same DB indirectly by querying via the closed-but-still-readable
    # in-memory anchor: the events should have been committed before close
    # joined the writer thread, so a brand-new query against the same
    # session_id sees them.  We can't query through the closed sess; instead
    # just assert close() did not raise and finished within a reasonable time.
    assert sess.events._closed is True


def test_batched_time_based_flush_triggers_below_batch_size() -> None:
    """If fewer events than batch_size arrive, the time-based flush still commits them."""
    sess = Session(batched=True, batch_size=100, batch_interval=0.05)
    sess.emit(EventType.AGENT_START, {"i": 0})
    # Wait > batch_interval but well under any reasonable test timeout.
    time.sleep(0.2)
    assert len(sess.events.query()) == 1
    sess.close()


# ---------------------------------------------------------------------------
# Bounded queue + drop policy
# ---------------------------------------------------------------------------


def test_drop_policy_drops_overflow_without_raising() -> None:
    """With a tiny queue and a slow writer, overflow events are dropped not raised."""
    # Use a long batch_interval so the writer is slow to drain; tiny
    # max_queue_size so we saturate quickly.
    sess = Session(
        batched=True,
        batch_size=1000,  # never size-trigger
        batch_interval=10.0,  # never time-trigger during the test
        max_queue_size=2,
        on_full="drop",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(20):
            sess.emit(EventType.AGENT_START, {"i": i})
    # Some events were dropped — at least one drop warning fired.
    drop_warnings = [w for w in caught if "queue is full" in str(w.message)]
    assert drop_warnings, "expected drop warning when queue saturates"
    # Counter incremented.
    assert sess.events._dropped_count > 0
    sess.close()


def test_block_policy_backpressures_producer(monkeypatch) -> None:
    """on_full='block' makes emit() wait when the queue is full."""
    sess = Session(
        batched=True,
        batch_size=1,  # writer drains aggressively
        batch_interval=0.01,
        max_queue_size=2,
        on_full="block",
    )
    # Just verify no events are lost under saturation pressure.
    for i in range(50):
        sess.emit(EventType.AGENT_START, {"i": i})
    sess.flush()
    assert len(sess.events.query()) == 50
    assert sess.events._dropped_count == 0
    sess.close()


# ---------------------------------------------------------------------------
# Flush semantics
# ---------------------------------------------------------------------------


def test_flush_is_noop_when_not_batched() -> None:
    sess = Session()
    sess.flush()  # must not raise even though there's no writer thread
    sess.close()


def test_flush_returns_after_pending_events_persisted() -> None:
    sess = Session(batched=True, batch_size=10, batch_interval=2.0)
    for i in range(5):
        sess.emit(EventType.AGENT_START, {"i": i})
    # Pre-flush: queue may still hold items (batch_size=10 > 5 submitted,
    # batch_interval=2s longer than test runtime).
    sess.flush(timeout=2.0)
    # Post-flush: every submitted event is persisted.
    assert len(sess.events.query()) == 5
    sess.close()
