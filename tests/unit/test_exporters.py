"""Tests for the pluggable event exporter system."""

import json
import pytest
from pathlib import Path

from lazybridge.exporters import (
    CallbackExporter,
    EventExporter,
    FilteredExporter,
    JsonFileExporter,
)
from lazybridge.lazy_session import EventLog, Event, LazySession, TrackLevel


# ---------------------------------------------------------------------------
# CallbackExporter
# ---------------------------------------------------------------------------

def test_callback_exporter_receives_events():
    """CallbackExporter receives events when registered on EventLog."""
    collected = []
    exp = CallbackExporter(collected.append)

    log = EventLog("sess-1", exporters=[exp])
    log.log(Event.TOOL_CALL, agent_id="a1", agent_name="worker", name="add", arguments={"a": 1})

    assert len(collected) == 1
    event = collected[0]
    assert event["event_type"] == "tool_call"
    assert event["agent_id"] == "a1"
    assert event["agent_name"] == "worker"
    assert event["data"]["name"] == "add"


def test_callback_exporter_rejects_non_callable():
    with pytest.raises(TypeError, match="callable"):
        CallbackExporter("not a function")


def test_multiple_exporters_all_receive():
    """All registered exporters receive every event."""
    c1, c2 = [], []
    log = EventLog("sess-1", exporters=[CallbackExporter(c1.append), CallbackExporter(c2.append)])
    log.log(Event.AGENT_START, method="loop", task="test")

    assert len(c1) == 1
    assert len(c2) == 1
    assert c1[0]["event_type"] == c2[0]["event_type"] == "agent_start"


def test_exporter_failure_does_not_crash_log():
    """A failing exporter doesn't prevent EventLog from working."""
    good_events = []

    def bad_export(event):
        raise RuntimeError("boom")

    log = EventLog(
        "sess-1",
        exporters=[CallbackExporter(bad_export), CallbackExporter(good_events.append)],
    )
    log.log(Event.MODEL_REQUEST, model="test", n_messages=3)

    # Second exporter still received the event
    assert len(good_events) == 1
    # EventLog's in-memory store still has the event
    events = log.get()
    assert len(events) == 1


def test_exporter_failure_preserves_sqlite_write(tmp_path):
    """Exporter failure doesn't break SQLite persistence."""
    db = str(tmp_path / "test.db")

    def bad_export(event):
        raise ValueError("exporter broke")

    log = EventLog("sess-1", db=db, exporters=[CallbackExporter(bad_export)])
    log.log(Event.TOOL_RESULT, name="calc", result="42")

    # SQLite should still have the event
    events = log.get()
    assert len(events) == 1
    assert events[0]["event_type"] == "tool_result"


# ---------------------------------------------------------------------------
# FilteredExporter
# ---------------------------------------------------------------------------

def test_filtered_exporter_passes_matching_events():
    collected = []
    inner = CallbackExporter(collected.append)
    filtered = FilteredExporter(inner, event_types={Event.TOOL_CALL, Event.TOOL_RESULT})

    log = EventLog("sess-1", exporters=[filtered])
    log.log(Event.MODEL_REQUEST, model="test", n_messages=1)
    log.log(Event.TOOL_CALL, name="add", arguments={})
    log.log(Event.MODEL_RESPONSE, model="test", content="hi")
    log.log(Event.TOOL_RESULT, name="add", result="3")

    assert len(collected) == 2
    assert collected[0]["event_type"] == "tool_call"
    assert collected[1]["event_type"] == "tool_result"


def test_filtered_exporter_blocks_non_matching():
    collected = []
    filtered = FilteredExporter(
        CallbackExporter(collected.append),
        event_types={"agent_start"},
    )
    log = EventLog("sess-1", exporters=[filtered])
    log.log(Event.TOOL_CALL, name="x", arguments={})
    assert len(collected) == 0


# ---------------------------------------------------------------------------
# JsonFileExporter
# ---------------------------------------------------------------------------

def test_json_file_exporter_writes_jsonl(tmp_path):
    path = str(tmp_path / "events.jsonl")
    exp = JsonFileExporter(path)

    log = EventLog("sess-1", exporters=[exp])
    log.log(Event.AGENT_START, method="loop", task="test task")
    log.log(Event.AGENT_FINISH, method="loop", stop_reason="end_turn", n_steps=3)

    lines = Path(path).read_text().strip().split("\n")
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    assert e1["event_type"] == "agent_start"
    e2 = json.loads(lines[1])
    assert e2["event_type"] == "agent_finish"


# ---------------------------------------------------------------------------
# TrackLevel gating
# ---------------------------------------------------------------------------

def test_off_level_skips_exporters():
    """When TrackLevel.OFF, exporters should NOT receive events."""
    collected = []
    log = EventLog("sess-1", level=TrackLevel.OFF, exporters=[CallbackExporter(collected.append)])
    log.log(Event.MODEL_REQUEST, model="test", n_messages=1)
    assert len(collected) == 0


def test_verbose_events_skip_exporters_at_basic_level():
    """Verbose-only events are filtered before reaching exporters."""
    collected = []
    log = EventLog("sess-1", level=TrackLevel.BASIC, exporters=[CallbackExporter(collected.append)])
    log.log(Event.MESSAGES, messages=[])  # verbose-only
    assert len(collected) == 0


# ---------------------------------------------------------------------------
# add_exporter / remove_exporter
# ---------------------------------------------------------------------------

def test_add_exporter_at_runtime():
    collected = []
    log = EventLog("sess-1")
    log.log(Event.MODEL_REQUEST, model="test", n_messages=1)  # before add
    assert len(collected) == 0

    log.add_exporter(CallbackExporter(collected.append))
    log.log(Event.MODEL_RESPONSE, model="test", content="hi")  # after add
    assert len(collected) == 1


def test_remove_exporter():
    collected = []
    exp = CallbackExporter(collected.append)
    log = EventLog("sess-1", exporters=[exp])
    log.log(Event.TOOL_CALL, name="x", arguments={})
    assert len(collected) == 1

    log.remove_exporter(exp)
    log.log(Event.TOOL_RESULT, name="x", result="done")
    assert len(collected) == 1  # no new event


# ---------------------------------------------------------------------------
# LazySession integration
# ---------------------------------------------------------------------------

def test_session_passes_exporters_to_eventlog():
    collected = []
    sess = LazySession(exporters=[CallbackExporter(collected.append)])
    sess.events.log(Event.AGENT_START, method="loop", task="test")
    assert len(collected) == 1


def test_session_add_exporter_convenience():
    collected = []
    sess = LazySession()
    sess.add_exporter(CallbackExporter(collected.append))
    sess.events.log(Event.MODEL_REQUEST, model="test", n_messages=1)
    assert len(collected) == 1


# ---------------------------------------------------------------------------
# EventExporter protocol
# ---------------------------------------------------------------------------

def test_protocol_compliance():
    """CallbackExporter satisfies the EventExporter protocol."""
    exp = CallbackExporter(lambda e: None)
    assert isinstance(exp, EventExporter)


def test_custom_exporter_protocol():
    """A custom class with export() satisfies the protocol."""
    class MyExporter:
        def export(self, event):
            pass

    assert isinstance(MyExporter(), EventExporter)
