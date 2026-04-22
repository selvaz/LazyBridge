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

from lazybridge.graph import GraphSchema


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
        # Include ``run_id`` so callers (e.g. usage_summary) don't have
        # to re-query the DB row-by-row just to resolve it.
        return [
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "run_id": r["run_id"],
                "payload": json.loads(r["payload"]),
                "ts": r["ts"],
            }
            for r in rows
        ]


class Session:
    """Container for observability config: exporters, redaction, EventLog."""

    def __init__(
        self,
        *,
        db: str | None = None,
        exporters: list[Any] | None = None,
        redact: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        console: bool = False,
    ) -> None:
        self.session_id = str(uuid.uuid4())
        self.events = EventLog(self.session_id, db=db)
        self._exporters: list[Any] = list(exporters or [])
        self._redact = redact
        self._lock = threading.Lock()
        self.graph = GraphSchema(session_id=self.session_id)
        if console:
            # Late import to avoid circular dependency with exporters
            from lazybridge.exporters import ConsoleExporter

            self._exporters.append(ConsoleExporter())

    # ------------------------------------------------------------------
    # Graph registration — called by Agent.__init__ when session= is set.
    # ------------------------------------------------------------------

    def register_agent(self, agent: Any) -> None:
        """Register an agent with this session's graph."""
        self.graph.add_agent(agent)

    def register_tool_edge(self, from_agent: Any, to_agent: Any, *, label: str = "") -> None:
        """Record a tool-call edge between two registered agents."""
        from_id = str(getattr(from_agent, "name", "agent"))
        to_id = str(getattr(to_agent, "name", "agent"))
        from lazybridge.graph import EdgeType

        self.graph.add_edge(from_id, to_id, label=label, kind=EdgeType.TOOL)

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
            # Validate the redactor's return: must stay a dict.  A
            # misbehaving redactor used to either crash downstream on
            # ``{**None}`` or silently wipe the payload.  Now we warn
            # once per redactor callable and fall back to the original
            # payload — observability stays honest.
            result = self._redact(payload)
            if isinstance(result, dict):
                payload = result
            else:
                import warnings

                if not getattr(self._redact, "_lazybridge_warned", False):
                    warnings.warn(
                        f"Session redact callable returned "
                        f"{type(result).__name__!s}; expected dict. "
                        f"Payload left unredacted.",
                        stacklevel=2,
                    )
                    try:
                        self._redact._lazybridge_warned = True  # type: ignore[attr-defined]
                    except AttributeError:
                        pass   # built-in / frozen callable — best-effort
        self.events.record(event_type, payload, run_id=run_id)
        exporters = self._exporters  # snapshot for thread safety
        event_dict = {"event_type": str(event_type), "session_id": self.session_id, "run_id": run_id, **payload}
        for exp in exporters:
            try:
                exp.export(event_dict)
            except Exception as exc:
                # Warn once per exporter instance so a buggy exporter
                # is visible in logs instead of silently eating events.
                if not getattr(exp, "_lazybridge_export_warned", False):
                    import warnings

                    warnings.warn(
                        f"Exporter {exp.__class__.__name__} raised "
                        f"{type(exc).__name__}: {exc}. Further failures "
                        f"from this exporter will be suppressed.",
                        stacklevel=2,
                    )
                    try:
                        exp._lazybridge_export_warned = True  # type: ignore[attr-defined]
                    except AttributeError:
                        pass

    def usage_summary(self) -> dict[str, Any]:
        """Aggregate token usage and cost across all agent runs in this session.

        Returns a dict with:
          - "total": {input_tokens, output_tokens, cost_usd}
          - "by_agent": {agent_name: {input_tokens, output_tokens, cost_usd}}
          - "by_run":   {run_id:    {agent_name, input_tokens, output_tokens, cost_usd}}

        Post-fix this is O(events) with TWO queries total
        (AGENT_START + MODEL_RESPONSE) instead of the pre-fix
        ``2 × N + 2`` pattern where every row triggered a single-row
        SELECT to resolve its ``run_id``.  ``EventLog.query`` now
        surfaces ``run_id`` directly in the result dict.
        """
        # Two bulk queries.  No per-row DB trip.
        agent_starts = self.events.query(event_type=EventType.AGENT_START)
        model_responses = self.events.query(event_type=EventType.MODEL_RESPONSE)

        # Build run_id → agent_name map.
        run_agent: dict[str, str] = {
            row["run_id"]: row["payload"].get("agent_name", "unknown")
            for row in agent_starts
            if row.get("run_id")
        }

        total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_agent: dict[str, dict[str, Any]] = {}
        by_run: dict[str, dict[str, Any]] = {}

        for row in model_responses:
            p = row["payload"]
            run_id = row.get("run_id")
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
