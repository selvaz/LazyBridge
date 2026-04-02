"""LazyStore — explicit shared blackboard for cross-agent, cross-loop state.

Agents write and read keyed values explicitly. This is the right mechanism for
sharing state between agents that don't know about each other (pattern C) or
that run in separate loops (pattern B).

For context injection into a system prompt use LazyContext (lazy_context.py).
For direct tool return values use the tool return value itself.

Two backends, same API:
  - InMemory (default): fast, process-local, lost on process exit
  - SQLite: persistent across runs (activated via LazySession(db="path.db"))
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

class StoreEntry:
    """A single value stored in LazyStore."""

    __slots__ = ("key", "value", "agent_id", "written_at")

    def __init__(self, key: str, value: Any, agent_id: str | None = None) -> None:
        self.key = key
        self.value = value
        self.agent_id = agent_id
        self.written_at = datetime.now(UTC)

    def __repr__(self) -> str:
        return f"StoreEntry(key={self.key!r}, agent_id={self.agent_id!r})"


# ---------------------------------------------------------------------------
# InMemory backend
# ---------------------------------------------------------------------------

class _InMemoryBackend:
    def __init__(self) -> None:
        self._data: dict[str, StoreEntry] = {}
        self._lock = threading.Lock()

    def write(self, key: str, value: Any, agent_id: str | None = None) -> None:
        with self._lock:
            self._data[key] = StoreEntry(key, value, agent_id)

    def read(self, key: str) -> Any:
        with self._lock:
            entry = self._data.get(key)
        return entry.value if entry else None

    def read_entry(self, key: str) -> StoreEntry | None:
        with self._lock:
            return self._data.get(key)

    def read_all(self) -> dict[str, Any]:
        with self._lock:
            return {k: e.value for k, e in self._data.items()}

    def read_by_agent(self, agent_id: str) -> dict[str, Any]:
        with self._lock:
            return {k: e.value for k, e in self._data.items() if e.agent_id == agent_id}

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def entries(self) -> list[StoreEntry]:
        with self._lock:
            return list(self._data.values())


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

class _SQLiteBackend:
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS store_entries (
        key        TEXT    PRIMARY KEY,
        value_json TEXT    NOT NULL,
        agent_id   TEXT,
        written_at TEXT    NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_store_agent ON store_entries (agent_id);
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(self._SCHEMA)

    @contextmanager
    def _conn(self):
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            # WAL mode allows concurrent readers + one writer; busy_timeout avoids SQLITE_BUSY errors
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=10000")
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    def write(self, key: str, value: Any, agent_id: str | None = None) -> None:
        now = datetime.now(UTC).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store_entries (key, value_json, agent_id, written_at)"
                " VALUES (?, ?, ?, ?)",
                (key, json.dumps(value), agent_id, now),
            )

    def read(self, key: str) -> Any:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value_json FROM store_entries WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def read_entry(self, key: str) -> StoreEntry | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT key, value_json, agent_id, written_at FROM store_entries WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        entry = StoreEntry(row[0], json.loads(row[1]), row[2])
        entry.written_at = datetime.fromisoformat(row[3])
        return entry

    def read_all(self) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value_json FROM store_entries"
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def read_by_agent(self, agent_id: str) -> dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value_json FROM store_entries WHERE agent_id = ?",
                (agent_id,),
            ).fetchall()
        return {row[0]: json.loads(row[1]) for row in rows}

    def keys(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key FROM store_entries").fetchall()
        return [r[0] for r in rows]

    def delete(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM store_entries WHERE key = ?", (key,))

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM store_entries")

    def entries(self) -> list[StoreEntry]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, value_json, agent_id, written_at FROM store_entries"
            ).fetchall()
        result = []
        for row in rows:
            e = StoreEntry(row[0], json.loads(row[1]), row[2])
            e.written_at = datetime.fromisoformat(row[3])
            result.append(e)
        return result


# ---------------------------------------------------------------------------
# LazyStore (public façade)
# ---------------------------------------------------------------------------

class LazyStore:
    """Shared key-value blackboard for agent pipelines.

    Usage::

        store = LazyStore()                         # in-memory
        store = LazyStore(db="pipeline.db")         # SQLite-backed

        store.write("findings", data, agent_id="researcher")
        store.read("findings")
        store.read_all()
        store.read_by_agent("researcher")
    """

    def __init__(self, db: str | None = None) -> None:
        self._backend: _InMemoryBackend | _SQLiteBackend = (
            _SQLiteBackend(db) if db else _InMemoryBackend()
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, key: str, value: Any, *, agent_id: str | None = None) -> None:
        """Write a value under ``key``.  Overwrites any existing entry.

        ``value`` must be JSON-serializable (str, int, float, bool, None,
        list, or dict with JSON-compatible contents).  Non-serializable objects
        are rejected so behaviour is consistent between in-memory and SQLite
        backends.
        """
        try:
            import json as _json
            _json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"LazyStore.write(): value for key {key!r} is not JSON-serializable "
                f"({type(value).__name__}). Store only JSON-compatible types."
            ) from exc
        self._backend.write(key, value, agent_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, key: str, default: Any = None) -> Any:
        """Return the value for ``key``, or ``default`` if absent."""
        entry = self._backend.read_entry(key)
        return default if entry is None else entry.value

    def read_entry(self, key: str) -> StoreEntry | None:
        """Return the full StoreEntry (includes agent_id, written_at)."""
        return self._backend.read_entry(key)

    def read_all(self) -> dict[str, Any]:
        """Return all entries as ``{key: value}``."""
        return self._backend.read_all()

    def read_by_agent(self, agent_id: str) -> dict[str, Any]:
        """Return all entries written by ``agent_id`` as ``{key: value}``."""
        return self._backend.read_by_agent(agent_id)

    def keys(self) -> list[str]:
        return self._backend.keys()

    def entries(self) -> list[StoreEntry]:
        """Return all StoreEntry objects (useful for GUI/serialization)."""
        return self._backend.entries()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def delete(self, key: str) -> None:
        self._backend.delete(key)

    def clear(self) -> None:
        self._backend.clear()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return self._backend.read_entry(key) is not None

    def __getitem__(self, key: str) -> Any:
        entry = self._backend.read_entry(key)
        if entry is None:
            raise KeyError(key)
        return entry.value

    def __setitem__(self, key: str, value: Any) -> None:
        self.write(key, value)

    def to_text(self, keys: list[str] | None = None) -> str:
        """Render store contents as a text block (for context injection).

        If ``keys`` is given only those keys are included.
        """
        data = self.read_all() if not keys else {k: self.read(k) for k in keys if k in self}
        if not data:
            return ""
        lines = ["[shared store]"]
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
