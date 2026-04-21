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

    def usage_summary(self) -> dict[str, Any]:
        """Aggregate token usage and cost across all agent runs in this session.

        Returns a dict with:
          - "total": {input_tokens, output_tokens, cost_usd}
          - "by_agent": {agent_name: {input_tokens, output_tokens, cost_usd}}
          - "by_run":   {run_id:    {agent_name, input_tokens, output_tokens, cost_usd}}
        """
        model_responses = self.events.query(event_type=EventType.MODEL_RESPONSE)
        agent_starts = {
            e["id"]: e["payload"]
            for e in self.events.query(event_type=EventType.AGENT_START)
        }

        # Build run_id → agent_name map from AGENT_START events
        run_agent: dict[str, str] = {}
        for row in self.events.query(event_type=EventType.AGENT_START):
            rid = row.get("payload", {}).get("run_id") or ""
            # run_id stored in the record table, not payload — fetch from raw
            raw = self.events._conn().execute(
                "SELECT run_id FROM events WHERE id=?", (row["id"],)
            ).fetchone()
            if raw and raw["run_id"]:
                run_agent[raw["run_id"]] = row["payload"].get("agent_name", "unknown")

        total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_agent: dict[str, dict[str, Any]] = {}
        by_run: dict[str, dict[str, Any]] = {}

        for row in model_responses:
            p = row["payload"]
            raw = self.events._conn().execute(
                "SELECT run_id FROM events WHERE id=?", (row["id"],)
            ).fetchone()
            run_id = raw["run_id"] if raw else None
            agent_name = run_agent.get(run_id or "", "unknown") if run_id else "unknown"

            in_tok = p.get("input_tokens", 0) or 0
            out_tok = p.get("output_tokens", 0) or 0
            cost = p.get("cost_usd") or 0.0

            total["input_tokens"] += in_tok
            total["output_tokens"] += out_tok
            total["cost_usd"] += cost

            ag = by_agent.setdefault(agent_name, {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            ag["input_tokens"] += in_tok
            ag["output_tokens"] += out_tok
            ag["cost_usd"] += cost

            if run_id:
                rn = by_run.setdefault(run_id, {"agent_name": agent_name, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                rn["input_tokens"] += in_tok
                rn["output_tokens"] += out_tok
                rn["cost_usd"] += cost

        return {"total": total, "by_agent": by_agent, "by_run": by_run}
