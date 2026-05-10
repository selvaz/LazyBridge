"""Regression test for B8 — OTel exporter must log exceptions, not swallow them.

Background: ``OTelExporter._end_span`` previously wrapped every SDK call
(``set_status``, ``context.detach``, ``span.end``) in ``try: ... except:
pass``.  A failure in any of those left a span half-written with no signal
to the operator.  The fix replaces the bare ``pass`` with a WARNING-level
log entry so the failure is visible in standard tooling.

This test injects faulty SDK objects to force each branch and asserts the
warning is emitted with a useful message.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_otel_set_status_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel import OTelExporter

    sink = InMemorySpanExporter()
    exp = OTelExporter(exporter=sink)

    # Drive a normal start, then poison the registered span so set_status raises.
    exp.export({"event_type": "agent_start", "run_id": "r1", "agent_name": "a"})

    run_spans = exp._spans["r1"]
    (entry,) = run_spans.values()
    entry.span.set_status = MagicMock(side_effect=RuntimeError("boom-set-status"))

    with caplog.at_level(logging.WARNING, logger="lazybridge.ext.otel.exporter"):
        exp.export({"event_type": "agent_error", "run_id": "r1", "agent_name": "a", "error": "x"})

    assert any("set_status" in rec.message and "boom-set-status" in rec.message for rec in caplog.records), (
        f"expected set_status warning, got: {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_otel_span_end_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from lazybridge.ext.otel import OTelExporter

    sink = InMemorySpanExporter()
    exp = OTelExporter(exporter=sink)
    exp.export({"event_type": "agent_start", "run_id": "r2", "agent_name": "a"})

    run_spans = exp._spans["r2"]
    (entry,) = run_spans.values()
    entry.span.end = MagicMock(side_effect=RuntimeError("boom-end"))

    with caplog.at_level(logging.WARNING, logger="lazybridge.ext.otel.exporter"):
        exp.export({"event_type": "agent_finish", "run_id": "r2", "agent_name": "a"})

    assert any("span.end" in rec.message and "boom-end" in rec.message for rec in caplog.records), (
        f"expected span.end warning, got: {[r.message for r in caplog.records]}"
    )
