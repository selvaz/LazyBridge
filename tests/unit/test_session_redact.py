"""Tests for ``Session(redact=..., redact_on_error=...)``.

Redactor failures (raising OR returning a non-dict) under
``redact_on_error="strict"`` (the default) drop the event entirely so
no unredacted data leaks; ``redact_on_error="fallback"`` warns once
and persists the original payload for callers who prefer lossless
event capture.

Also covers the default-safe secret redactor: when ``Session()`` is
constructed without ``redact=``, well-known credential shapes inside
event payloads are stripped before they reach the EventLog or any
exporter.  ``unsafe_log_payloads=True`` and the explicit
``redact=None`` opt-out are tested separately.
"""

from __future__ import annotations

import warnings

import pytest

from lazybridge.session import EventType, Session, redact_secrets


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


# ---------------------------------------------------------------------------
# Default-safe secret redactor — ``redact_secrets``
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_label",
    [
        # OpenAI / Anthropic style (sk- prefix, 20+ chars)
        ("sk-abc123XYZ456abc123XYZ", "openai-style-key"),
        ("sk-ant-abc123XYZ456abc123XYZ", "openai-style-key"),
        # Stripe live key
        ("sk_live_abcdefghij0123456789", "stripe-key"),
        # GitHub PAT (ghp_ + 36 chars)
        ("ghp_" + "a" * 36, "github-token"),
        # Google API key (AIza + 35 chars)
        ("AIza" + "B" * 35, "google-api-key"),
        # Slack bot token
        ("xoxb-1234567890-abcdef", "slack-token"),
        # JWT
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature_part", "jwt"),
    ],
)
def test_redact_secrets_masks_known_credential_shapes(raw: str, expected_label: str) -> None:
    out = redact_secrets({"tool_result": f"the api key is {raw} thanks"})
    assert raw not in out["tool_result"]
    assert f"<redacted:{expected_label}>" in out["tool_result"]


def test_redact_secrets_masks_bearer_authorization_header() -> None:
    out = redact_secrets({"req": "Authorization: Bearer abcdefghijklmnop"})
    # Bearer keyword is preserved so the credential's presence stays
    # visible in the trace even after the value is stripped.
    assert "Bearer <redacted:bearer>" in out["req"]
    assert "abcdefghijklmnop" not in out["req"]


def test_redact_secrets_leaves_innocent_prose_alone() -> None:
    payload = {
        "task": "Summarise the quarterly report.",
        "result": "Revenue rose 12% — see attached.",
        "uuid": "550e8400-e29b-41d4-a716-446655440000",
    }
    assert redact_secrets(payload) == payload


def test_redact_secrets_recurses_into_nested_structures() -> None:
    payload = {
        "outer": {
            "messages": [
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": "use key sk-" + "z" * 24},
            ]
        }
    }
    out = redact_secrets(payload)
    assistant_content = out["outer"]["messages"][1]["content"]
    assert "sk-" + "z" * 24 not in assistant_content
    assert "<redacted:openai-style-key>" in assistant_content


# ---------------------------------------------------------------------------
# Session default — secret redactor is wired by default
# ---------------------------------------------------------------------------


def test_session_default_redacts_secrets_without_explicit_redact_kwarg() -> None:
    sess = Session()  # no redact= passed → default secret redactor
    sess.emit(
        EventType.TOOL_RESULT,
        {"step": "lookup", "result": "received sk-" + "Q" * 24 + " from upstream"},
    )
    rows = sess.events.query()
    assert "<redacted:openai-style-key>" in rows[0]["payload"]["result"]


def test_session_unsafe_log_payloads_disables_default_redaction() -> None:
    sess = Session(unsafe_log_payloads=True)
    raw = "received sk-" + "Q" * 24 + " from upstream"
    sess.emit(EventType.TOOL_RESULT, {"step": "lookup", "result": raw})
    rows = sess.events.query()
    # Explicit opt-out: the original credential survives in the log.
    assert rows[0]["payload"]["result"] == raw


def test_session_explicit_redact_none_disables_default_redaction() -> None:
    sess = Session(redact=None)
    raw = "received sk-" + "Q" * 24 + " from upstream"
    sess.emit(EventType.TOOL_RESULT, {"step": "lookup", "result": raw})
    rows = sess.events.query()
    assert rows[0]["payload"]["result"] == raw


def test_session_explicit_redact_callable_overrides_default() -> None:
    def my_redactor(p: dict) -> dict:
        # User redactor that does *not* mask sk- keys — the default
        # must not be stacked in front of a user-supplied callable.
        return {**p, "marker": "mine"}

    sess = Session(redact=my_redactor)
    raw = "sk-" + "Q" * 24
    sess.emit(EventType.TOOL_RESULT, {"step": "lookup", "result": raw, "marker": ""})
    rows = sess.events.query()
    # User redactor ran (marker rewritten) and default did NOT
    # (sk- key still visible).
    assert rows[0]["payload"]["marker"] == "mine"
    assert raw in rows[0]["payload"]["result"]
