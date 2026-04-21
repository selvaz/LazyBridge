"""Session — event bus, SQLite-backed EventLog, and observability container."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    AGENT_START = "agent_start"
    AGENT_FINISH = "agent_finish"
    LOOP_STEP = "loop_step"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"


class EventLog:
    """SQLite-backed event log. Thread-safe via thread-local connections."""

    def __init__(self, session_id: str, db: str | None = None) -> None:
        self.session_id = session_id
        self._db = db
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            if self._db:
                self._local.conn = sqlite3.connect(self._db, check_same_thread=False)
                self._local.conn.execute("PRAGMA journal_mode=WAL")
                self._local.conn.execute("PRAGMA busy_timeout=5000")
            else:
                self._local.conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        self._conn().execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                run_id TEXT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                ts REAL NOT NULL
            )
            """
        )
        self._conn().commit()

    def record(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        row = {
            "session_id": self.session_id,
            "run_id": run_id,
            "event_type": str(event_type),
            "payload": payload,
            "ts": time.time(),
        }
        self._conn().execute(
            "INSERT INTO events (session_id, run_id, event_type, payload, ts) VALUES (?,?,?,?,?)",
            (row["session_id"], row["run_id"], row["event_type"], json.dumps(row["payload"]), row["ts"]),
        )
        self._conn().commit()
        return row

    def query(self, *, run_id: str | None = None, event_type: EventType | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE session_id=?"
        params: list[Any] = [self.session_id]
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        if event_type:
            sql += " AND event_type=?"
            params.append(str(event_type))
        sql += " ORDER BY id ASC"
        rows = self._conn().execute(sql, params).fetchall()
        return [{"id": r["id"], "event_type": r["event_type"], "payload": json.loads(r["payload"]), "ts": r["ts"]} for r in rows]


class Session:
    """Container for observability config: exporters, redaction, EventLog."""

    def __init__(
        self,
        *,
        db: str | None = None,
        exporters: list[Any] | None = None,
        redact: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.events = EventLog(self.session_id, db=db)
        self._exporters: list[Any] = list(exporters or [])
        self._redact = redact
        self._lock = threading.Lock()

    def add_exporter(self, exporter: Any) -> None:
        with self._lock:
            self._exporters = [*self._exporters, exporter]

    def remove_exporter(self, exporter: Any) -> None:
        with self._lock:
            self._exporters = [e for e in self._exporters if e is not exporter]

    def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> None:
        if self._redact:
            payload = self._redact(payload)
        self.events.record(event_type, payload, run_id=run_id)
        exporters = self._exporters  # snapshot for thread safety
        event_dict = {"event_type": str(event_type), "session_id": self.session_id, "run_id": run_id, **payload}
        for exp in exporters:
            try:
                exp.export(event_dict)
            except Exception:
                pass
