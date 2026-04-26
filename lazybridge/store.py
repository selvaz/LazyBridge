"""Store — thread-safe key-value blackboard with optional SQLite persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any


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
        # ``RLock`` (not ``Lock``) so internal helpers like
        # ``_discard_thread_conn`` can safely re-enter the lock when
        # called from a code path (``compare_and_swap``) that already
        # holds it.
        self._lock = threading.RLock()
        # Track every opened thread-local connection so we can close
        # them deterministically from ``close()``.  ``threading.local``
        # only scopes attributes per thread; without a registry the
        # connections linger until the owning thread exits + GC runs,
        # which leaks file descriptors under worker pools.
        self._all_conns: list[sqlite3.Connection] = []
        self._closed = False
        if not db:
            self._mem: dict[str, StoreEntry] = {}
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        if not self._db:
            # Every public method branches on ``self._db`` before it
            # ever reaches here; this arm is only hit if a future
            # refactor wires a SQLite call into the in-memory path.
            # Fail fast with a real error instead of handing back
            # ``None`` behind a ``type: ignore``.
            raise RuntimeError("Store._conn called in in-memory mode — use the _mem path.")
        if self._closed:
            raise RuntimeError("Store is closed")
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self._db, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._lock:
                self._all_conns.append(conn)
        return self._local.conn

    def _discard_thread_conn(self) -> None:
        """Drop this thread's cached SQLite connection.

        Used after a transaction recovery fails (ROLLBACK itself
        raised) so a future call on this thread doesn't inherit a
        connection in an undefined transaction state.  The connection
        is removed from ``_all_conns`` and best-effort closed; the
        next ``_conn()`` call rebuilds.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            return
        try:
            del self._local.conn
        except AttributeError:
            pass
        with self._lock:
            try:
                self._all_conns.remove(conn)
            except ValueError:
                pass
        try:
            conn.close()
        except sqlite3.Error:
            pass

    def close(self) -> None:
        """Close every thread-local SQLite connection opened by this Store.

        Idempotent.  After ``close()`` the Store raises ``RuntimeError``
        on further reads / writes so callers fail fast instead of
        silently re-opening connections.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            conns = list(self._all_conns)
            self._all_conns.clear()
        for c in conns:
            try:
                c.close()
            except sqlite3.Error:
                pass  # already closed / invalid — nothing to recover

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        # Best-effort finalizer.  Python doesn't guarantee __del__ runs,
        # so users relying on deterministic cleanup should call close()
        # or use the context manager.
        try:
            self.close()
        except Exception:
            pass

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
        # Pydantic instances are serialised via ``model_dump(mode="json")``
        # before JSON encoding — otherwise SQLite storage would fall
        # through ``default=str`` and persist the model's ``__repr__``
        # (e.g. ``"x=42 name='hello'"``), which is NOT round-trippable.
        # In-memory storage keeps the instance as-is since Python dicts
        # don't need JSON round-tripping.
        if self._db:
            serialised = _to_jsonable(value)
            self._conn().execute(
                "INSERT OR REPLACE INTO store (key, value, written_at, agent_id) VALUES (?,?,?,?)",
                (key, json.dumps(serialised, default=str), time.time(), agent_id),
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
            return StoreEntry(
                key=row["key"], value=json.loads(row["value"]), written_at=row["written_at"], agent_id=row["agent_id"]
            )
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

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, key: object) -> bool:
        return key in self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    def compare_and_swap(
        self,
        key: str,
        expected: Any,
        new: Any,
        *,
        agent_id: str | None = None,
    ) -> bool:
        """Atomically set ``key`` to ``new`` only if its current value
        deep-equals ``expected``.

        ``expected=None`` means "the key must not currently exist" (a
        missing key compares equal to ``None``).

        Returns ``True`` on success, ``False`` when another writer has
        moved the value since the caller's last read — the caller is
        expected to treat this as a lost race (raise, back off, re-read,
        etc.).  Comparison uses the JSON-normalised shape (via
        :func:`_to_jsonable`), so Pydantic models / nested dicts compare
        the same as they do on disk and survive SQLite round-trips.

        SQLite path: wraps the read-check-write in ``BEGIN IMMEDIATE``
        so concurrent writers serialise on the SQLite reserved lock
        instead of interleaving.
        """
        if self._closed:
            raise RuntimeError("Store is closed")
        if not self._db:
            with self._lock:
                entry = self._mem.get(key)
                cur = entry.value if entry else None
                if not _json_eq(cur, expected):
                    return False
                self._mem[key] = StoreEntry(key=key, value=new, agent_id=agent_id)
                return True

        # SQLite path — serialise concurrent writers via reserved lock.
        conn = self._conn()
        serialised_new = json.dumps(_to_jsonable(new), default=str)
        with self._lock:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT value FROM store WHERE key=?", (key,)).fetchone()
                cur = json.loads(row["value"]) if row else None
                if not _json_eq(cur, expected):
                    conn.execute("ROLLBACK")
                    return False
                conn.execute(
                    "INSERT OR REPLACE INTO store (key, value, written_at, agent_id) VALUES (?,?,?,?)",
                    (key, serialised_new, time.time(), agent_id),
                )
                conn.commit()
                return True
            except sqlite3.Error:
                # Inner ROLLBACK can itself fail (e.g. the connection
                # was already torn down by the OS).  If that happens,
                # the connection is in an undefined state — discard
                # this thread's cached connection so the next caller
                # gets a fresh one instead of inheriting the poison.
                # Surface the rollback failure as a warning so an
                # operator can notice persistent transaction issues.
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error as rollback_exc:
                    import warnings as _warnings

                    _warnings.warn(
                        f"Store.compare_and_swap: ROLLBACK after error failed "
                        f"({type(rollback_exc).__name__}: {rollback_exc}).  "
                        f"Discarding the thread-local connection so the next "
                        f"call gets a fresh one.",
                        UserWarning,
                        stacklevel=3,
                    )
                    self._discard_thread_conn()
                raise

    def to_text(self, keys: list[str] | None = None) -> str:
        data = self.read_all()
        if keys:
            data = {k: v for k, v in data.items() if k in keys}
        return "\n".join(f"{k}: {json.dumps(v, default=str)}" for k, v in data.items())


def _json_eq(a: Any, b: Any) -> bool:
    """Deep-equality under the same JSON shape the Store persists.

    Compare values after ``_to_jsonable`` so a Pydantic model passed in
    at runtime matches the dict it round-trips to on disk.  Falls back
    to ``False`` if either side isn't JSON-serialisable (rather than
    raising — a non-serialisable expected value can't have been written
    by this Store, so CAS should simply miss).
    """
    try:
        sa = json.dumps(_to_jsonable(a), sort_keys=True, default=str)
        sb = json.dumps(_to_jsonable(b), sort_keys=True, default=str)
    except TypeError:
        return False
    return sa == sb


def _to_jsonable(value: Any) -> Any:
    """Convert a value to a JSON-serialisable shape.

    Handles Pydantic v2 ``BaseModel`` via ``model_dump(mode="json")``;
    recurses into lists / tuples / dicts so a ``list[BaseModel]`` also
    round-trips.  Plain primitives are returned unchanged.
    """
    try:
        from pydantic import BaseModel
    except ImportError:  # pragma: no cover
        BaseModel = None  # type: ignore[misc, assignment]

    if BaseModel is not None and isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value
