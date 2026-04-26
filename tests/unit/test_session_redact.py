"""Tests for ``Session(redact=..., redact_on_error=...)``.

Redactor failures (raising OR returning a non-dict) under
``redact_on_error="strict"`` (the default) drop the event entirely so
no unredacted data leaks; ``redact_on_error="fallback"`` warns once
and persists the original payload for callers who prefer lossless
event capture.
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge.session import EventType, Session


def _emit_sample(sess: Session) -> None:
    sess.emit(
        EventType.AGENT_START,
        {"agent": "alice", "task": "x", "ssn": "123-45-6789"},
    )


# ---------------------------------------------------------------------------
# Happy path — redactor works
# ---------------------------------------------------------------------------


def test_well_behaved_redactor_masks_payload() -> None:
    def redact(p: dict) -> dict:
        return {**p, "ssn": "***"}

    sess = Session(redact=redact)
    _emit_sample(sess)
    rows = sess.events.query()
    assert rows[0]["payload"]["ssn"] == "***"


# ---------------------------------------------------------------------------
# Fallback mode (default)
# ---------------------------------------------------------------------------


def test_redactor_raising_fallback_records_unredacted_and_warns() -> None:
    def bad(_p: dict) -> dict:
        raise RuntimeError("upstream vault down")

    sess = Session(redact=bad, redact_on_error="fallback")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _emit_sample(sess)

    rows = sess.events.query()
    # Event IS recorded even though redactor raised.
    assert len(rows) == 1
    assert rows[0]["payload"]["ssn"] == "123-45-6789"  # unredacted
    # Warning fired.
    assert any("raised" in str(w.message) for w in caught)


def test_redactor_non_dict_return_fallback_records_unredacted_and_warns() -> None:
    def wrong(_p: dict) -> None:  # type: ignore[return]
        return None

    sess = Session(redact=wrong, redact_on_error="fallback")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _emit_sample(sess)

    rows = sess.events.query()
    assert len(rows) == 1
    assert rows[0]["payload"]["ssn"] == "123-45-6789"
    assert any("expected dict" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Strict mode — event is dropped
# ---------------------------------------------------------------------------


def test_redactor_raising_strict_drops_event() -> None:
    def bad(_p: dict) -> dict:
        raise RuntimeError("vault offline")

    sess = Session(redact=bad, redact_on_error="strict")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _emit_sample(sess)

    rows = sess.events.query()
    # No event recorded — unredacted payload must not reach the log.
    assert rows == []
    # Warning still fired so operators can see the redactor is broken.
    assert any("raised" in str(w.message) for w in caught)


def test_redactor_non_dict_strict_drops_event() -> None:
    def wrong(_p: dict) -> str:  # type: ignore[return]
        return "not a dict"

    sess = Session(redact=wrong, redact_on_error="strict")
    _emit_sample(sess)
    assert sess.events.query() == []


def test_strict_mode_also_skips_exporters() -> None:
    """Strict = no unredacted data on the bus at all — neither log
    nor exporters see the event when the redactor fails."""
    seen: list[dict] = []

    class _Capture:
        def export(self, evt: dict) -> None:
            seen.append(evt)

    def bad(_p: dict) -> dict:
        raise RuntimeError("no-op")

    sess = Session(
        redact=bad,
        redact_on_error="strict",
        exporters=[_Capture()],
    )
    _emit_sample(sess)
    assert seen == []


def test_strict_mode_still_works_on_well_behaved_redactor() -> None:
    """Strict is only a failure-mode flag — when the redactor works,
    the redacted payload flows through exactly as in fallback mode."""

    def redact(p: dict) -> dict:
        return {**p, "ssn": "***"}

    sess = Session(redact=redact, redact_on_error="strict")
    _emit_sample(sess)
    rows = sess.events.query()
    assert rows[0]["payload"]["ssn"] == "***"


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_invalid_redact_on_error_rejected() -> None:
    with pytest.raises(ValueError, match="redact_on_error"):
        Session(redact_on_error="queue")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Warning de-duplication — once per (redactor, failure mode)
# ---------------------------------------------------------------------------


def test_warning_emits_once_per_failure_mode() -> None:
    def bad(_p: dict) -> dict:
        raise RuntimeError("boom")

    sess = Session(redact=bad, redact_on_error="fallback")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(5):
            _emit_sample(sess)

    raise_warnings = [w for w in caught if "raised" in str(w.message)]
    # Single warning despite five events — the redactor is stamped.
    assert len(raise_warnings) == 1
    # All five events still recorded (fallback mode).
    assert len(sess.events.query()) == 5
