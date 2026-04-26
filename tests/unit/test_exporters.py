"""Tests for the v1.0 exporter system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from lazybridge.exporters import (
    CallbackExporter,
    EventExporter,
    FilteredExporter,
    JsonFileExporter,
    StructuredLogExporter,
)

_SAMPLE_EVENT = {"event_type": "tool_call", "tool": "search", "run_id": "r1"}


# ── EventExporter protocol ────────────────────────────────────────────────────


def test_event_exporter_protocol():
    assert isinstance(CallbackExporter(lambda e: None), EventExporter)
    assert isinstance(JsonFileExporter("/tmp/x.jsonl"), EventExporter)


# ── CallbackExporter ──────────────────────────────────────────────────────────


def test_callback_receives_event():
    received = []
    exp = CallbackExporter(received.append)
    exp.export(_SAMPLE_EVENT)
    assert len(received) == 1
    assert received[0]["event_type"] == "tool_call"


def test_callback_multiple_events():
    received = []
    exp = CallbackExporter(received.append)
    for i in range(5):
        exp.export({"event_type": "loop_step", "turn": i})
    assert len(received) == 5


def test_callback_exception_in_fn_propagates():
    def _boom(e):
        raise ValueError("intentional")

    exp = CallbackExporter(_boom)
    with pytest.raises(ValueError, match="intentional"):
        exp.export(_SAMPLE_EVENT)


# ── FilteredExporter ──────────────────────────────────────────────────────────


def test_filtered_passes_matching():
    received = []
    inner = CallbackExporter(received.append)
    exp = FilteredExporter(inner, event_types={"tool_call", "tool_result"})
    exp.export({"event_type": "tool_call"})
    exp.export({"event_type": "model_request"})
    assert len(received) == 1
    assert received[0]["event_type"] == "tool_call"


def test_filtered_drops_non_matching():
    received = []
    exp = FilteredExporter(CallbackExporter(received.append), event_types={"agent_start"})
    exp.export({"event_type": "tool_call"})
    assert len(received) == 0


def test_filtered_empty_set_drops_all():
    received = []
    exp = FilteredExporter(CallbackExporter(received.append), event_types=set())
    exp.export({"event_type": "anything"})
    assert len(received) == 0


# ── JsonFileExporter ──────────────────────────────────────────────────────────


def test_json_file_appends_lines():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="r", delete=False) as f:
        path = f.name

    exp = JsonFileExporter(path)
    exp.export({"event_type": "agent_start", "agent": "test"})
    exp.export({"event_type": "agent_finish"})

    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event_type"] == "agent_start"


def test_json_file_handles_non_serializable():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    import datetime

    exp = JsonFileExporter(path)
    exp.export({"event_type": "tool_result", "ts": datetime.datetime.now()})
    lines = Path(path).read_text().strip().splitlines()
    assert len(lines) == 1  # did not raise


# ── StructuredLogExporter ─────────────────────────────────────────────────────


def test_structured_log_emits(caplog):
    import logging

    exp = StructuredLogExporter("lazybridge.test")
    with caplog.at_level(logging.INFO, logger="lazybridge.test"):
        exp.export({"event_type": "model_response", "content": "hello"})
    assert any("model_response" in r.message for r in caplog.records)
