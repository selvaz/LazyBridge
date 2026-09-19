"""The control store: schema, atomic claim, fencing, and the ledger write.

Everything that changes state goes through one connection, one
transaction, and leaves one event behind.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Long enough that a normal writer never sees it, short enough that a
#: genuine deadlock surfaces as an error rather than a hang.
BUSY_TIMEOUT_SECONDS = 30.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id   TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'draft',
    created_at   REAL NOT NULL
);

-- Who may see a project. Absence is denial: a caller with no row here
-- cannot read the project, and the repository enforces that rather than
-- trusting a caller to filter. A UI filter is a display choice; this is
-- the boundary.
CREATE TABLE IF NOT EXISTS project_assignments (
    project_id   TEXT NOT NULL,
    actor_id     TEXT NOT NULL,
    PRIMARY KEY (project_id, actor_id)
);

CREATE TABLE IF NOT EXISTS queue_items (
    item_id      TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    status       TEXT NOT NULL,           -- ready | claimed | done | failed
    owner        TEXT,
    fence        INTEGER NOT NULL DEFAULT 0,
    lease_expires_at REAL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS queue_ready ON queue_items(status, created_at);

-- Append-only. Nothing here is ever updated or deleted: the state of an
-- item is derivable from its events, so a row that disagrees with its
-- history is a visible contradiction instead of the only surviving copy.
CREATE TABLE IF NOT EXISTS run_events (
    event_id     TEXT PRIMARY KEY,
    item_id      TEXT NOT NULL,
    project_id   TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    actor_id     TEXT,
    fence        INTEGER,
    detail       TEXT,
    occurred_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_item ON run_events(item_id, occurred_at);
"""


class ScopeDenied(PermissionError):
    """The actor may not see or touch this project."""


class FenceRejected(RuntimeError):
    """A worker reported on a claim it no longer holds.

    Raised rather than returned: a caller that ignores this is a caller
    writing a result nobody is waiting for, and silence there is how two
    writers for one item stay invisible.
    """


@dataclass(frozen=True)
class ClaimedItem:
    item_id: str
    project_id: str
    payload: dict[str, Any]
    fence: int
    attempts: int


class ControlStore:
    """One database, opened per process, safe for concurrent processes."""

    def __init__(self, path: str | Path, *, lease_seconds: float = 900.0) -> None:
        # A lease of 0 lets a second claimant take an item the moment the
        # first has it; a negative one is the same; NaN compares false
        # against every clock reading, so the claimed item could never be
        # reclaimed. None of those is a configuration, they are mistakes
        # that would surface as duplicate work or a stuck queue.
        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or not (math.isfinite(lease_seconds) and lease_seconds > 0)
        ):
            raise ValueError(f"lease_seconds must be a positive, finite number of seconds, got {lease_seconds!r}")
        self.path = str(path)
        self.lease_seconds = lease_seconds
        self._conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL so readers never block the writer; NORMAL because FULL costs a
        # disk sync per commit for a durability guarantee we do not need
        # against process death (only against machine power loss, which for
        # this fleet means "the run is gone anyway").
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_SECONDS * 1000)}")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """A write transaction that takes the lock up front.

        BEGIN IMMEDIATE, not the default deferred begin: a deferred
        transaction that reads first and writes later can be aborted with
        SQLITE_BUSY halfway through, after the caller has already made its
        decision on what it read. Taking the write lock at the start turns
        that race into a wait.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    # --- events ---------------------------------------------------------

    def _record(
        self,
        conn: sqlite3.Connection,
        *,
        item_id: str,
        project_id: str,
        event_type: str,
        actor_id: str | None = None,
        fence: int | None = None,
        detail: str | None = None,
        now: float,
    ) -> None:
        conn.execute(
            "INSERT INTO run_events (event_id, item_id, project_id, event_type, actor_id, fence, detail, occurred_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), item_id, project_id, event_type, actor_id, fence, detail, now),
        )

    # --- projects -------------------------------------------------------

    def create_project(self, project_id: str, title: str, *, status: str = "draft") -> None:
        now = time.time()
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO projects (project_id, title, status, created_at) VALUES (?, ?, ?, ?)",
                (project_id, title, status, now),
            )

    def assign(self, project_id: str, actor_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO project_assignments (project_id, actor_id) VALUES (?, ?)",
                (project_id, actor_id),
            )

    def _assert_scope(self, conn: sqlite3.Connection, project_id: str, actor_id: str | None) -> None:
        if actor_id is None:
            return  # an unscoped caller is the control plane itself
        row = conn.execute(
            "SELECT 1 FROM project_assignments WHERE project_id=? AND actor_id=?",
            (project_id, actor_id),
        ).fetchone()
        if row is None:
            # Deliberately the same message whether the project is absent
            # or merely not visible: distinguishing them tells an actor
            # that a project it cannot see exists, which is half of what
            # it wanted to know.
            raise ScopeDenied(f"actor {actor_id!r} is not assigned to project {project_id!r}")

    def visible_projects(self, actor_id: str | None) -> list[str]:
        if actor_id is None:
            rows = self._conn.execute("SELECT project_id FROM projects ORDER BY project_id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT p.project_id FROM projects p"
                " JOIN project_assignments a ON a.project_id = p.project_id"
                " WHERE a.actor_id = ? ORDER BY p.project_id",
                (actor_id,),
            ).fetchall()
        return [row["project_id"] for row in rows]

    # --- queue ----------------------------------------------------------

    def enqueue(self, project_id: str, payload: dict[str, Any], *, actor_id: str | None = None) -> str:
        item_id = str(uuid.uuid4())
        now = time.time()
        with self._write() as conn:
            self._assert_scope(conn, project_id, actor_id)
            # A queue item for a project that does not exist is work nobody
            # can see: it never appears in visible_projects, so it is
            # outside the scope model entirely while still being claimable.
            # A typo in a project id should not be able to create that.
            # Found by Codex review on PR #173.
            if conn.execute("SELECT 1 FROM projects WHERE project_id=?", (project_id,)).fetchone() is None:
                raise ScopeDenied(f"no project {project_id!r} -- create it before queueing work for it")
            conn.execute(
                "INSERT INTO queue_items (item_id, project_id, payload, status, created_at, updated_at)"
                " VALUES (?, ?, ?, 'ready', ?, ?)",
                (item_id, project_id, json.dumps(payload), now, now),
            )
            self._record(
                conn,
                item_id=item_id,
                project_id=project_id,
                event_type="queued",
                actor_id=actor_id,
                now=now,
            )
        return item_id

    def claim(self, owner: str, *, actor_id: str | None = None, now: float | None = None) -> ClaimedItem | None:
        """Take the oldest available item, or None.

        The whole method is one write transaction. A ready item, or one
        whose lease has lapsed, becomes this owner's with the fence
        incremented -- so the previous holder's token is now stale and its
        late report will be refused.
        """
        moment = now if now is not None else time.time()
        with self._write() as conn:
            if actor_id is None:
                row = conn.execute(
                    "SELECT * FROM queue_items WHERE status='ready'"
                    " OR (status='claimed' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)"
                    " ORDER BY created_at LIMIT 1",
                    (moment,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT q.* FROM queue_items q"
                    " JOIN project_assignments a ON a.project_id = q.project_id AND a.actor_id = ?"
                    " WHERE q.status='ready'"
                    " OR (q.status='claimed' AND q.lease_expires_at IS NOT NULL AND q.lease_expires_at <= ?)"
                    " ORDER BY q.created_at LIMIT 1",
                    (actor_id, moment),
                ).fetchone()
            if row is None:
                return None
            fence = int(row["fence"]) + 1
            conn.execute(
                "UPDATE queue_items SET status='claimed', owner=?, fence=?, lease_expires_at=?,"
                " attempts=attempts+1, updated_at=? WHERE item_id=?",
                (owner, fence, moment + self.lease_seconds, moment, row["item_id"]),
            )
            self._record(
                conn,
                item_id=row["item_id"],
                project_id=row["project_id"],
                event_type="claimed",
                actor_id=owner,
                fence=fence,
                now=moment,
            )
            return ClaimedItem(
                item_id=row["item_id"],
                project_id=row["project_id"],
                payload=json.loads(row["payload"]),
                fence=fence,
                attempts=int(row["attempts"]) + 1,
            )

    def finish(self, item_id: str, *, fence: int, status: str, detail: str | None = None) -> None:
        """Close a claim. Refuses a stale fence."""
        if status not in ("done", "failed"):
            raise ValueError("status must be 'done' or 'failed'")
        now = time.time()
        with self._write() as conn:
            row = conn.execute("SELECT * FROM queue_items WHERE item_id=?", (item_id,)).fetchone()
            if row is None:
                raise FenceRejected(f"no such item {item_id!r}")
            if int(row["fence"]) != fence:
                raise FenceRejected(
                    f"item {item_id!r} is at fence {row['fence']}, not {fence} -- "
                    "this claim was taken over while the worker was away"
                )
            if row["status"] in ("done", "failed"):
                raise FenceRejected(f"item {item_id!r} already reached {row['status']!r}")
            if row["status"] != "claimed":
                # Without this, finish(item, fence=0) right after enqueue
                # succeeds: a fence of 0 matches a never-claimed row, so
                # unprocessed work is removed from the queue, a terminal
                # event is recorded with no claim before it, and the
                # ledger accepts the result as clean. Found by Codex
                # review on PR #173.
                raise FenceRejected(
                    f"item {item_id!r} is {row['status']!r}, not claimed -- "
                    "work cannot be completed before it has been taken"
                )
            conn.execute(
                "UPDATE queue_items SET status=?, lease_expires_at=NULL, updated_at=? WHERE item_id=?",
                (status, now, item_id),
            )
            self._record(
                conn,
                item_id=item_id,
                project_id=row["project_id"],
                event_type=status,
                actor_id=row["owner"],
                fence=fence,
                detail=detail,
                now=now,
            )

    # --- reads ----------------------------------------------------------

    def item(self, item_id: str, *, actor_id: str | None = None) -> dict[str, Any] | None:
        """One item, scoped.

        The scope check belongs here and not only on the write paths: an
        item id is a UUID, but a UUID is not a secret, and a caller that
        obtains one from a log or a shared report would otherwise read
        another project's payload through a method that looks harmless.
        Found by Codex review on PR #173.
        """
        row = self._conn.execute("SELECT * FROM queue_items WHERE item_id=?", (item_id,)).fetchone()
        if row is None:
            if actor_id is not None:
                # Same answer as "not yours". Returning None to a scoped
                # actor for an absent id, and raising for a present one,
                # is an oracle for which ids exist -- and, worse for
                # events(), it skipped the check entirely when the item row
                # was gone but its events remained.
                raise ScopeDenied(f"actor {actor_id!r} may not read item {item_id!r}")
            return None
        if actor_id is not None:
            try:
                self._assert_scope(self._conn, row["project_id"], actor_id)
            except ScopeDenied:
                raise ScopeDenied(f"actor {actor_id!r} may not read item {item_id!r}") from None
        return dict(row)

    def events(self, item_id: str, *, actor_id: str | None = None) -> list[dict[str, Any]]:
        if actor_id is not None:
            # Raises rather than returning an empty list: a silent empty
            # answer is indistinguishable from "this item has no history",
            # and a caller cannot tell it was refused. The denial is the
            # information.
            self.item(item_id, actor_id=actor_id)
        rows = self._conn.execute(
            "SELECT * FROM run_events WHERE item_id=? ORDER BY occurred_at, rowid", (item_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    @contextmanager
    def _consistent_read(self) -> Iterator[None]:
        """One read transaction: every SELECT inside sees the same committed state.

        The ledger reads events and items in two statements. In autocommit
        those are two snapshots, so a worker finishing an item between them
        makes verification see the events from before and the row from
        after -- and report a healthy item as ``terminal_without_event``.

        Two edges, both found by Codex review of the first version:

        - **Already inside a transaction.** An unconditional ``BEGIN`` raises
          "cannot start a transaction within a transaction" and leaves the
          caller's transaction open. The ambient one already gives a
          consistent view, so this joins it and neither begins nor ends
          anything.
        - **Ending it.** A read has nothing to commit, so it is ended with
          ``ROLLBACK``. A ``COMMIT`` in a ``finally`` could raise its own
          error and replace the real one -- the failed read that the caller
          actually needs to see. On the error path the rollback is
          best-effort and the original exception propagates untouched.
        """
        if self._conn.in_transaction:
            yield
            return
        self._conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        else:
            self._conn.execute("ROLLBACK")

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT status, COUNT(*) n FROM queue_items GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    # Bulk reads, for the ledger check and the operator CLI -- which ARE
    # the control plane. Underscored rather than given an actor argument:
    # nothing else has a reason to enumerate, and a scoped variant nobody
    # calls is surface to keep correct for free.
    def _every_item(self) -> Sequence[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM queue_items").fetchall()]

    def _every_event(self) -> Sequence[dict[str, Any]]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM run_events ORDER BY occurred_at, rowid").fetchall()]
