"""Store — thread-safe key-value blackboard with optional SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StoreEntry:
    key: str
    value: Any
    written_at: float = field(default_factory=time.time)
    agent_id: str | None = None


class Store:
    """Key-value store for PlanState and shared data.

    db=None   → in-memory (lost on exit).
    db="file" → SQLite with WAL mode (persistent).
    """

    def __init__(self, db: str | None = None) -> None:
        self._db = db
        self._local = threading.local()
        self._lock = threading.Lock()
        if not db:
            self._mem: dict[str, StoreEntry] = {}
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not self._db:
            return None  # type: ignore[return-value]
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db, check_same_thread=False)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        if not self._db:
            return
        self._conn().execute(
            """
            CREATE TABLE IF NOT EXISTS store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                written_at REAL NOT NULL,
                agent_id TEXT
            )
            """
        )
        self._conn().commit()

    def write(self, key: str, value: Any, *, agent_id: str | None = None) -> None:
        if self._db:
            self._conn().execute(
                "INSERT OR REPLACE INTO store (key, value, written_at, agent_id) VALUES (?,?,?,?)",
                (key, json.dumps(value, default=str), time.time(), agent_id),
            )
            self._conn().commit()
        else:
            with self._lock:
                self._mem[key] = StoreEntry(key=key, value=value, agent_id=agent_id)

    def read(self, key: str, default: Any = None) -> Any:
        if self._db:
            row = self._conn().execute("SELECT value FROM store WHERE key=?", (key,)).fetchone()
            return json.loads(row["value"]) if row else default
        with self._lock:
            entry = self._mem.get(key)
            return entry.value if entry else default

    def read_entry(self, key: str) -> StoreEntry | None:
        if self._db:
            row = self._conn().execute("SELECT * FROM store WHERE key=?", (key,)).fetchone()
            if not row:
                return None
            return StoreEntry(key=row["key"], value=json.loads(row["value"]), written_at=row["written_at"], agent_id=row["agent_id"])
        with self._lock:
            return self._mem.get(key)

    def read_all(self) -> dict[str, Any]:
        if self._db:
            rows = self._conn().execute("SELECT key, value FROM store").fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}
        with self._lock:
            return {k: v.value for k, v in self._mem.items()}

    def delete(self, key: str) -> None:
        if self._db:
            self._conn().execute("DELETE FROM store WHERE key=?", (key,))
            self._conn().commit()
        else:
            with self._lock:
                self._mem.pop(key, None)

    def clear(self) -> None:
        if self._db:
            self._conn().execute("DELETE FROM store")
            self._conn().commit()
        else:
            with self._lock:
                self._mem.clear()

    def keys(self) -> list[str]:
        if self._db:
            rows = self._conn().execute("SELECT key FROM store").fetchall()
            return [r["key"] for r in rows]
        with self._lock:
            return list(self._mem.keys())

    def to_text(self, keys: list[str] | None = None) -> str:
        data = self.read_all()
        if keys:
            data = {k: v for k, v in data.items() if k in keys}
        return "\n".join(f"{k}: {json.dumps(v, default=str)}" for k, v in data.items())
