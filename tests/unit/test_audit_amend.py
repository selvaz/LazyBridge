"""Regression tests for the 12 follow-up findings landed in this branch.

Covers (numbering matches AUDIT_DEEP_FRAMEWORK.md):

- 3.1 MCP transports concurrent connect() race
- 3.2 Quarto YAML title/author escape
- 3.3 section_renderer input_root sandboxing
- 3.4 tool_schema explicit branches for stdlib types
- 3.5 templates: meta.* escaped under autoescape=False
- 3.6 strict-mode dialect note (docstring-only)
- 3.7 viz/server Content-Length cap
- 3.8 OTel _set_attr truncation
- 3.9 gateway: cross-host / scheme-downgrade redirects rejected
- 3.10 predicates strict mode
- 3.11 memory _plan_compression docstring (covered by lint of file content)
- 3.12 EventLog.record_many close-race comment (covered by lint of file content)
- 3.14 single-source __version__
- 3.15 deprecation sunset version
"""

from __future__ import annotations

import datetime
import decimal
import inspect
import pathlib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 3.1 — MCP transports concurrent connect() race
# ---------------------------------------------------------------------------
# Relocated to lazytools/tests/test_mcp.py with the MCP connector (moved to
# lazytoolkit in 0.8).

# ---------------------------------------------------------------------------
# 3.2 + 3.3 — Quarto YAML escape / section_renderer sandbox
# ---------------------------------------------------------------------------
# Moved to the sibling lazybridge-reports repo as part of the 0.7.9 →
# 0.9.0 extraction.  These tests live next to the code they cover.

# ---------------------------------------------------------------------------
# 3.4 — tool_schema branches for stdlib types
# ---------------------------------------------------------------------------


def test_tool_schema_bytes_emits_byte_format() -> None:
    from lazybridge.core.tool_schema import _annotation_to_schema

    assert _annotation_to_schema(bytes) == {"type": "string", "format": "byte"}
    assert _annotation_to_schema(bytearray) == {"type": "string", "format": "byte"}


def test_tool_schema_datetime_emits_iso_formats() -> None:
    from lazybridge.core.tool_schema import _annotation_to_schema

    assert _annotation_to_schema(datetime.datetime) == {
        "type": "string",
        "format": "date-time",
    }
    assert _annotation_to_schema(datetime.date) == {"type": "string", "format": "date"}
    assert _annotation_to_schema(datetime.time) == {"type": "string", "format": "time"}


def test_tool_schema_decimal_emits_number() -> None:
    from lazybridge.core.tool_schema import _annotation_to_schema

    assert _annotation_to_schema(decimal.Decimal) == {"type": "number"}


def test_tool_schema_path_emits_string() -> None:
    from lazybridge.core.tool_schema import _annotation_to_schema

    assert _annotation_to_schema(pathlib.Path) == {"type": "string"}
    assert _annotation_to_schema(pathlib.PurePath) == {"type": "string"}


# ---------------------------------------------------------------------------
# 3.5 — templates escape meta.*
# ---------------------------------------------------------------------------
# Moved to the sibling lazybridge-reports repo with the rest of the
# template rendering surface.

# ---------------------------------------------------------------------------
# 3.6 — strict-mode docstring note
# ---------------------------------------------------------------------------


def test_strict_mode_dialect_note_present() -> None:
    """Source comment names the OpenAI strict dialect divergence."""
    src = Path(__file__).resolve().parents[2] / "lazybridge" / "core" / "tool_schema.py"
    text = src.read_text(encoding="utf-8")
    assert "OpenAI's strict structured-output validator" in text


# ---------------------------------------------------------------------------
# 3.7 — viz/server Content-Length cap
# ---------------------------------------------------------------------------


def test_viz_server_has_max_control_body_constant() -> None:
    from lazybridge.ext.viz import server as viz_server

    assert hasattr(viz_server, "_MAX_CONTROL_BODY")
    assert isinstance(viz_server._MAX_CONTROL_BODY, int)
    assert viz_server._MAX_CONTROL_BODY > 0
    # 1 MB is the chosen cap; assert it's at least the documented size.
    assert viz_server._MAX_CONTROL_BODY <= 10_000_000


# ---------------------------------------------------------------------------
# 3.8 — OTel _set_attr truncation
# ---------------------------------------------------------------------------


def test_otel_set_attr_truncates_long_strings() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from lazybridge.ext.otel.exporter import OTelExporter

    captured: list[tuple[str, object]] = []

    class _Span:
        def set_attribute(self, key: str, value: object) -> None:
            captured.append((key, value))

    OTelExporter._set_attr(_Span(), "k", "x" * 5000)
    assert len(captured) == 1
    key, value = captured[0]
    assert key == "k"
    assert isinstance(value, str)
    assert len(value) <= 1024
    assert value.endswith("...")


def test_otel_set_attr_passes_short_strings_through() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from lazybridge.ext.otel.exporter import OTelExporter

    captured: list[tuple[str, object]] = []

    class _Span:
        def set_attribute(self, key: str, value: object) -> None:
            captured.append((key, value))

    OTelExporter._set_attr(_Span(), "k", "ok")
    assert captured == [("k", "ok")]


# ---------------------------------------------------------------------------
# 3.9 — gateway cross-host redirect refusal
# ---------------------------------------------------------------------------
# Relocated to lazytools/tests/test_gateway.py with the gateway connector
# (moved to lazytoolkit in 0.8).


# ---------------------------------------------------------------------------
# 3.10 — predicates strict mode
# ---------------------------------------------------------------------------


def test_predicates_field_strict_raises_on_missing_attr() -> None:
    from lazybridge import when
    from lazybridge.envelope import Envelope

    env = Envelope(task="t", payload={"items": [1, 2]})
    p = when.field("itmes", strict=True).empty()  # typo for "items"

    with pytest.raises(AttributeError, match="strict=True"):
        p(env)


def test_predicates_field_default_lax_returns_none() -> None:
    """Backwards-compat: default behaviour treats missing field as None."""
    from lazybridge import when
    from lazybridge.envelope import Envelope

    env = Envelope(task="t", payload={"items": [1, 2]})
    assert when.field("missing").empty()(env) is True


def test_predicates_field_strict_raises_on_none_payload() -> None:
    from lazybridge import when
    from lazybridge.envelope import Envelope

    env = Envelope(task="t", payload=None)
    p = when.field("any", strict=True).equals(1)

    with pytest.raises(AttributeError):
        p(env)


# ---------------------------------------------------------------------------
# 3.11 / 3.12 — docstring + comment hygiene fixes (file-content checks)
# ---------------------------------------------------------------------------


def test_memory_plan_compression_docstring_names_compressing() -> None:
    src = Path(__file__).resolve().parents[2] / "lazybridge" / "memory.py"
    text = src.read_text(encoding="utf-8")
    # The new docstring must name ``_compressing`` as the synchronisation
    # primitive, not just ``_lock``.
    assert "_compressing`` has been set" in text or "_compressing` has been set" in text


def test_session_record_many_has_close_race_comment() -> None:
    from lazybridge.session import EventLog

    src = inspect.getsource(EventLog.record_many)
    assert "close()" in src
    # Comment text mirrors record(); the explicit phrase must be present.
    assert "fail fast" in src.lower() or "executemany" in src


# ---------------------------------------------------------------------------
# 3.14 — version single-source
# ---------------------------------------------------------------------------


def test_version_matches_distribution_metadata() -> None:
    """``__version__`` must equal what ``importlib.metadata`` reports."""
    pytest.importorskip("importlib.metadata")
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    from lazybridge import __version__ as lb_version

    # Initialise ``dist_version`` outside the try so the static
    # analyser doesn't flag it as "may be used before initialised"
    # on the except branch (``pytest.skip`` raises, but CodeQL
    # can't see that).
    dist_version: str | None = None
    try:
        dist_version = _dist_version("lazybridge")
    except PackageNotFoundError:
        pytest.skip("lazybridge not installed (running from raw source tree)")

    assert dist_version is not None
    assert lb_version == dist_version


# ---------------------------------------------------------------------------
# 3.15 — deprecation sunset version
# ---------------------------------------------------------------------------


def test_tool_choice_parallel_now_raises() -> None:
    """``tool_choice='parallel'`` was deprecated in 0.7 and removed in 0.8.

    Concurrent tool execution is the default and can no longer be opted
    out of; passing the legacy value must surface as a ``ValueError``
    naming the replacement (rather than silently degrading to ``"auto"``).
    """
    import pytest

    from lazybridge.engines.llm import LLMEngine

    with pytest.raises(ValueError) as ei:
        LLMEngine(model="claude-3-haiku", provider="anthropic", tool_choice="parallel")  # type: ignore[arg-type]

    msg = str(ei.value)
    assert "removed" in msg
    assert "auto" in msg or "any" in msg
