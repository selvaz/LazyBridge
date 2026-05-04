"""Public API: live mode wiring + replay constructor."""

from __future__ import annotations

import http.client
import json

import pytest

from lazybridge.ext.viz import Visualizer
from lazybridge.session import EventType, Session


def test_live_mode_wires_exporter_and_serves(tmp_path):
    sess = Session()
    try:
        viz = Visualizer(sess, auto_open=False)
        viz.start()
        # Emit one event and read it via /api/snapshot
        sess.emit(EventType.AGENT_START, {"agent_name": "a"})
        # Allow the synchronous SQLite write to complete
        c = http.client.HTTPConnection(viz._server.host, viz._server.port, timeout=2.0)
        c.request("GET", f"/api/snapshot?t={viz._server.token}")
        r = c.getresponse()
        body = json.loads(r.read())
        assert r.status == 200
        seqs = [e.get("_seq") for e in body.get("events", [])]
        assert seqs and all(isinstance(s, int) for s in seqs)
        viz.stop()
    finally:
        sess.close()


def test_replay_mode_requires_existing_db(tmp_path):
    db = tmp_path / "empty.db"
    # Create an empty events table
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY, session_id TEXT, run_id TEXT, "
        "event_type TEXT, payload TEXT, ts REAL)"
    )
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="No sessions"):
        Visualizer.replay(str(db))


def test_replay_mode_loads_events_and_serves(tmp_path):
    db = tmp_path / "rec.db"
    sess = Session(db=str(db))
    sess.emit(EventType.AGENT_START, {"agent_name": "alice"})
    sess.emit(EventType.TOOL_CALL, {"agent_name": "alice", "name": "search", "arguments": {"q": "x"}})
    sess.flush()
    sess.close()

    viz = Visualizer.replay(str(db), auto_open=False)
    viz.start()
    c = http.client.HTTPConnection(viz._server.host, viz._server.port, timeout=2.0)
    c.request("GET", f"/api/meta?t={viz._server.token}")
    r = c.getresponse()
    meta = json.loads(r.read())
    assert meta["mode"] == "replay"
    assert meta["session_id"]
    viz.stop()
