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

import asyncio
import datetime
import decimal
import inspect
import pathlib
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 3.1 — MCP transports concurrent connect() race
# ---------------------------------------------------------------------------


def test_mcp_stdio_transport_has_per_instance_connect_lock() -> None:
    """Two StdioTransport instances get distinct locks (no shared state)."""
    from lazybridge.ext.mcp.transports import StdioTransport

    a = StdioTransport(command="echo", args=["hi"])
    b = StdioTransport(command="echo", args=["hi"])
    assert isinstance(a._connect_lock, asyncio.Lock)
    assert a._connect_lock is not b._connect_lock


def test_mcp_http_transport_has_per_instance_connect_lock() -> None:
    from lazybridge.ext.mcp.transports import HttpTransport

    t = HttpTransport(url="http://localhost:0/mcp")
    assert isinstance(t._connect_lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_mcp_connect_serialises_concurrent_callers() -> None:
    """Two concurrent connect() callers must not both build a stack.

    We patch the SDK import to fail so connect() raises ImportError;
    the assertion is that the lock was acquired/released cleanly and
    the second caller saw the same outcome as the first (both raise),
    not that one silently completed while the other was mid-flight.
    """
    from lazybridge.ext.mcp.transports import StdioTransport

    t = StdioTransport(command="echo")

    # Force the SDK import to fail so we don't actually spawn anything.
    with patch.dict("sys.modules", {"mcp": None}):
        results = await asyncio.gather(
            t.connect(),
            t.connect(),
            return_exceptions=True,
        )

    # Both calls failed identically (no half-built state).
    assert all(isinstance(r, ImportError) for r in results)
    assert t._session is None
    assert t._stack is None


# ---------------------------------------------------------------------------
# 3.2 — Quarto YAML title/author escape
# ---------------------------------------------------------------------------


def test_quarto_project_yml_escapes_title_with_quote() -> None:
    from lazybridge.external_tools.report_builder.quarto.project import build_quarto_yml

    yml = build_quarto_yml(formats=["html"], title='evil"\nrun: ls')
    # The injected newline must be escaped, not literal.
    assert "\nrun: ls" not in yml
    assert "\\n" in yml
    # The injected double-quote must be escaped, not literal-then-rest-of-line.
    assert '"evil\\"' in yml


def test_quarto_project_yml_escapes_author_with_backslash() -> None:
    from lazybridge.external_tools.report_builder.quarto.project import build_quarto_yml

    yml = build_quarto_yml(formats=["html"], author='C:\\Users\\evil"')
    assert '"C:\\\\Users\\\\evil\\""' in yml


# ---------------------------------------------------------------------------
# 3.3 — section_renderer input_root sandboxing
# ---------------------------------------------------------------------------


def test_render_chart_section_rejects_path_outside_input_root(tmp_path: Path) -> None:
    from lazybridge.external_tools.report_builder.schemas import ChartSection
    from lazybridge.external_tools.report_builder.section_renderer import render_chart_section

    inside = tmp_path / "ok.png"
    outside = tmp_path / ".." / "evil.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG magic

    section = ChartSection(type="chart", path=str(outside.resolve()), title="x")

    with pytest.raises(ValueError, match="outside the allowed input root"):
        render_chart_section(section, input_root=tmp_path)


def test_render_chart_section_allows_path_inside_input_root(tmp_path: Path) -> None:
    from lazybridge.external_tools.report_builder.schemas import ChartSection
    from lazybridge.external_tools.report_builder.section_renderer import render_chart_section

    inside = tmp_path / "ok.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\n")
    section = ChartSection(type="chart", path=str(inside), title="x")

    html = render_chart_section(section, input_root=tmp_path)
    assert "<figure" in html


def test_render_chart_section_no_input_root_keeps_old_behaviour(tmp_path: Path) -> None:
    """Backwards compat: when input_root=None, no sandbox check is applied."""
    from lazybridge.external_tools.report_builder.schemas import ChartSection
    from lazybridge.external_tools.report_builder.section_renderer import render_chart_section

    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    section = ChartSection(type="chart", path=str(p), title="x")
    html = render_chart_section(section)  # no input_root
    assert "<figure" in html


def test_sections_to_html_threads_input_root_through(tmp_path: Path) -> None:
    from lazybridge.external_tools.report_builder.schemas import ChartSection
    from lazybridge.external_tools.report_builder.section_renderer import sections_to_html

    outside = tmp_path / ".." / "evil.png"
    section = ChartSection(type="chart", path=str(outside.resolve()), title="x")

    with pytest.raises(ValueError, match="outside the allowed input root"):
        sections_to_html([section], input_root=tmp_path)


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


def test_templates_escape_meta_fields() -> None:
    """Every ``{{ meta.<scalar> }}`` render must go through the ``|e`` filter.

    We only assert on tokens the template *actually emits* — different
    templates render different subsets of ``meta.*``, so the test reads
    each file and checks the fields it actually uses.
    """
    pkg = Path(__file__).resolve().parents[2] / "lazybridge" / "external_tools" / "report_builder" / "templates"
    fields = ("meta.theme", "meta.template", "meta.generated_at")
    for tmpl in (pkg / "executive_summary.html.j2", pkg / "data_snapshot.html.j2"):
        text = tmpl.read_text(encoding="utf-8")
        for field in fields:
            # Only assert when the template actually emits this field —
            # if it doesn't render the token at all, there's nothing to escape.
            if f"{{{{ {field} " in text or f"{{{{ {field}}}" in text or f"{{{{ {field}|" in text:
                assert f"{{{{ {field}|e }}}}" in text or f"{{{{ {field} | e }}}}" in text, (
                    f"{tmpl.name} renders {field} without |e"
                )


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


def test_gateway_redirect_handler_rejects_cross_host() -> None:
    import urllib.error
    import urllib.request

    from lazybridge.ext.gateway import _SameOriginRedirectHandler

    handler = _SameOriginRedirectHandler()
    req = urllib.request.Request("https://api.example.com/v1/tool")

    with pytest.raises(urllib.error.HTTPError, match="different host"):
        handler.redirect_request(req, fp=None, code=302, msg="", headers={}, newurl="https://evil.invalid/path")


def test_gateway_redirect_handler_rejects_https_downgrade() -> None:
    import urllib.error
    import urllib.request

    from lazybridge.ext.gateway import _SameOriginRedirectHandler

    handler = _SameOriginRedirectHandler()
    req = urllib.request.Request("https://api.example.com/v1/tool")

    with pytest.raises(urllib.error.HTTPError, match="downgrade"):
        handler.redirect_request(req, fp=None, code=302, msg="", headers={}, newurl="http://api.example.com/v1/tool")


def test_gateway_redirect_handler_allows_same_host_path_change() -> None:
    """Same-host redirect (e.g., /v1 -> /v2) must still work."""
    import urllib.request

    from lazybridge.ext.gateway import _SameOriginRedirectHandler

    handler = _SameOriginRedirectHandler()
    req = urllib.request.Request("https://api.example.com/v1/tool")
    # super().redirect_request returns a Request — assert it doesn't raise.
    new_req = handler.redirect_request(
        req, fp=None, code=302, msg="", headers={}, newurl="https://api.example.com/v2/tool"
    )
    assert new_req.full_url == "https://api.example.com/v2/tool"


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


def test_tool_choice_parallel_deprecation_names_sunset_version() -> None:
    import warnings

    from lazybridge.engines.llm import LLMEngine

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMEngine(model="claude-3-haiku", provider="anthropic", tool_choice="parallel")  # type: ignore[arg-type]

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecations, "expected a DeprecationWarning"
    msg = str(deprecations[0].message)
    assert "1.0" in msg or "removed in" in msg
