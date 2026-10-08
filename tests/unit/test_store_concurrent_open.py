"""Several connections opening the same fresh SQLite Store at once."""

from __future__ import annotations

import threading
from pathlib import Path

from lazybridge import Store


def _open_together(path: Path, count: int = 4) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(count)

    def opener() -> None:
        barrier.wait()
        try:
            store = Store(db=str(path))
            store.write("k", {"v": 1})
            assert store.read("k") == {"v": 1}
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=opener) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_concurrent_first_open_does_not_hit_database_is_locked(tmp_path: Path) -> None:
    """The switch to WAL bypasses SQLite's busy handler; without a retry, a few
    Stores opening one fresh file together failed with 'database is locked'
    most of the time."""
    for trial in range(40):
        errors = _open_together(tmp_path / f"s{trial}.sqlite")
        assert errors == [], errors
