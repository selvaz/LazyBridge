"""Store.write/delete/clear/write_memory/delete_memory must recover from a
failed SQLite write the same way compare_and_swap already did -- otherwise
the thread-local connection stays pinned inside an open transaction and
every later read on that connection is frozen to a stale WAL snapshot for
the rest of the process's life. Confirmed live (see
lazyceo.approvals._fresh_read_store's docstring, and this exact fix
landing independently in LazyCEO before this one did).

Reproduced here with REAL lock contention (a second connection holding a
BEGIN EXCLUSIVE transaction), not a mock -- this is what "database is
locked" under concurrent writers actually looks like.
"""

from __future__ import annotations

import sqlite3

import pytest

from lazybridge.store import Store


def _make_contended_store(tmp_path) -> tuple[Store, sqlite3.Connection, str]:
    """A Store with a cached connection whose busy_timeout is dropped to
    50ms (so a lock-contention test fails fast instead of waiting the
    default 5s), plus a second raw connection holding an EXCLUSIVE lock."""
    db = str(tmp_path / "test.sqlite")
    store = Store(db=db)
    store.write("seed", "value")  # establishes store's cached connection
    store._conn().execute("PRAGMA busy_timeout=50")

    blocker = sqlite3.connect(db)
    blocker.execute("BEGIN EXCLUSIVE")
    return store, blocker, db


def test_write_recovers_after_real_lock_contention(tmp_path) -> None:
    store, blocker, db = _make_contended_store(tmp_path)

    with pytest.raises(sqlite3.OperationalError):
        store.write("k2", "v2")

    blocker.commit()
    blocker.close()

    other = Store(db=db)
    other.write("k3", "v3")
    other.close()

    # If the failed write above had left store's cached connection pinned
    # inside an open transaction, this read would still see the snapshot
    # from before k3 was written by the other connection.
    assert store.read("k3") == "v3"
    store.close()


def test_delete_recovers_after_real_lock_contention(tmp_path) -> None:
    store, blocker, db = _make_contended_store(tmp_path)

    with pytest.raises(sqlite3.OperationalError):
        store.delete("seed")

    blocker.commit()
    blocker.close()

    other = Store(db=db)
    other.write("k3", "v3")
    other.close()

    assert store.read("k3") == "v3"
    store.close()


def test_compare_and_swap_still_recovers_after_the_shared_helper_refactor(tmp_path) -> None:
    """compare_and_swap already had rollback-on-failure before this change
    -- confirm its behavior is unchanged now that it shares
    _recover_after_failed_write with write/delete/clear."""
    store, blocker, db = _make_contended_store(tmp_path)

    with pytest.raises(sqlite3.OperationalError):
        store.compare_and_swap("seed", "value", "new-value")

    blocker.commit()
    blocker.close()

    other = Store(db=db)
    other.write("k3", "v3")
    other.close()

    assert store.read("k3") == "v3"
    store.close()
