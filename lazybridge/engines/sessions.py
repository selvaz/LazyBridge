"""Durable session aliases for the coding engines.

``CodexEngine`` and ``ClaudeCodeEngine`` already resume a conversation by its
*native* id (``thread_id=`` / ``session_id=``), but those ids are opaque
UUIDs that a caller has to store somewhere and that an operator cannot
recognise, rename or re-point. :class:`SessionRegistry` is that somewhere: a
small JSON file mapping a human-chosen **alias** to the native id (plus the
model and effort it last ran with), so ``CodexEngine(session_alias="review")``
resumes the same Codex thread from any later process.

The registry is deliberately dumb — a file, not a service — and is meant to be
edited by hand when needed::

    {
      "version": 1,
      "sessions": {
        "codex": {
          "c:/work/repo": {
            "review": {"native_id": "019a...", "model": "gpt-6-luna",
                       "effort": "low", "created_at": "...", "updated_at": "..."}
          }
        }
      }
    }

An entry is keyed by ``(kind, scope, name)``. ``kind`` is ``"codex"`` or
``"claude"``; ``scope`` is a caller-chosen string, normally the engine's
working directory, so the same alias in two different projects is two
different sessions; ``name`` is the alias itself.

Concurrency: every mutation is a read-modify-write under an in-process lock
*and* an exclusive lock file next to the registry, and the file is replaced
atomically, so two threads or two processes binding at once cannot lose each
other's entries or leave a half-written file. No third-party dependency.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import threading
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "SessionRegistry",
    "default_session_registry",
    "normalize_scope",
    "set_default_session_registry",
]

#: Environment variable naming the registry file when no path is passed.
SESSIONS_FILE_ENV = "LAZYBRIDGE_SESSIONS_FILE"

_KINDS = frozenset({"codex", "claude"})
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
_FORMAT_VERSION = 1

#: A lock file older than this is presumed left behind by a crashed process.
#: A write takes milliseconds, so this is generous.
_LOCK_STALE_SECONDS = 30.0
_LOCK_TIMEOUT_SECONDS = 15.0
#: ``os.replace`` / reads on Windows fail with ``PermissionError`` while
#: another process has the target open for a moment: retry briefly.
_IO_RETRIES = 60
_IO_RETRY_SLEEP = 0.02


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_scope(scope: str | os.PathLike[str]) -> str:
    """Canonical form of a scope: resolved absolute posix path.

    Lower-cased on Windows, where paths are case-insensitive, so ``C:\\Work``
    and ``c:/work`` name the same scope. The scope is always treated as a
    path (a non-path string is resolved against the current directory), so
    pass a real directory, normally the engine's ``cwd``.
    """
    text = os.fspath(scope)
    if not text or not text.strip():
        raise ValueError("scope must be a non-empty string (normally the engine's cwd)")
    resolved = Path(text).resolve().as_posix()
    return resolved.lower() if os.name == "nt" else resolved


def _check_kind(kind: str) -> str:
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    return kind


def _check_label(name: str, what: str = "session alias") -> str:
    if not isinstance(name, str) or not _LABEL_RE.match(name):
        raise ValueError(
            f"invalid {what} {name!r}: must start with a letter and contain only letters, digits, "
            "'_', '.' or '-' (max 64 characters)"
        )
    return name


def _valid_shape(data: Any) -> bool:
    """Is ``data`` a registry document this module can work with?"""
    if not isinstance(data, dict):
        return False
    sessions = data.get("sessions")
    if not isinstance(sessions, dict):
        return False
    for by_scope in sessions.values():
        if not isinstance(by_scope, dict):
            return False
        for by_name in by_scope.values():
            if not isinstance(by_name, dict):
                return False
            for entry in by_name.values():
                if not isinstance(entry, dict) or not isinstance(entry.get("native_id"), str):
                    return False
    return True


class _CorruptRegistry(Exception):
    """Internal: the file exists but is not a usable registry document."""


class SessionRegistry:
    """JSON file store of named engine sessions.

    Parameters
    ----------
    path:
        Registry file. Defaults to the ``LAZYBRIDGE_SESSIONS_FILE``
        environment variable, else ``~/.lazybridge/sessions.json``. Resolved
        on every operation, not once, so changing the variable takes effect
        for a registry created without an explicit path.

    A missing file is an empty registry. A file that is not valid registry
    JSON is **never overwritten**: it is renamed to ``<name>.corrupt`` (or
    ``<name>.corrupt-<timestamp>`` when that already exists) and the registry
    starts empty.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path_arg = Path(path) if path is not None else None
        # Re-entrant: public methods call each other under the lock.
        self._tlock = threading.RLock()

    # -- location ----------------------------------------------------------

    @property
    def path(self) -> Path:
        """The file this registry reads and writes right now."""
        if self._path_arg is not None:
            return self._path_arg
        env = os.environ.get(SESSIONS_FILE_ENV)
        if env:
            return Path(env)
        return Path.home() / ".lazybridge" / "sessions.json"

    def __repr__(self) -> str:
        return f"SessionRegistry(path={str(self.path)!r})"

    # -- public API --------------------------------------------------------

    def resolve(self, kind: str, scope: str | os.PathLike[str], name: str) -> str | None:
        """The native id bound to ``(kind, scope, name)``, or ``None``."""
        kind, scope_key, name = _check_kind(kind), normalize_scope(scope), _check_label(name)
        entry = self._load()["sessions"].get(kind, {}).get(scope_key, {}).get(name)
        return entry["native_id"] if entry else None

    def bind(
        self,
        kind: str,
        scope: str | os.PathLike[str],
        name: str,
        native_id: str,
        *,
        model: str | None = None,
        effort: str | None = None,
    ) -> None:
        """Create the alias, or re-point an existing one at ``native_id``.

        ``model`` / ``effort`` record what the session last ran with; ``None``
        means "unspecified / the account default" and overwrites a previous
        value. ``created_at`` survives a rebind, ``updated_at`` does not.
        """
        kind, scope_key, name = _check_kind(kind), normalize_scope(scope), _check_label(name)
        if not isinstance(native_id, str) or not native_id.strip():
            raise ValueError("native_id must be a non-empty string")

        def mutate(sessions: dict[str, Any]) -> bool:
            by_name = sessions.setdefault(kind, {}).setdefault(scope_key, {})
            now = _now()
            previous = by_name.get(name)
            by_name[name] = {
                "native_id": native_id,
                "model": model,
                "effort": effort,
                "created_at": previous.get("created_at", now) if previous else now,
                "updated_at": now,
            }
            return True

        self._mutate(mutate)

    def rename(self, kind: str, scope: str | os.PathLike[str], old: str, new: str) -> None:
        """Rename an alias. ``KeyError`` if ``old`` is unknown, ``ValueError``
        if ``new`` already exists (an existing alias is never overwritten)."""
        kind, scope_key = _check_kind(kind), normalize_scope(scope)
        old, new = _check_label(old), _check_label(new, "new session alias")

        def mutate(sessions: dict[str, Any]) -> bool:
            by_name = sessions.get(kind, {}).get(scope_key, {})
            if old not in by_name:
                raise KeyError(f"no {kind} session alias {old!r} in scope {scope_key!r}")
            if old == new:
                return False
            if new in by_name:
                raise ValueError(f"{kind} session alias {new!r} already exists in scope {scope_key!r}; forget it first")
            entry = by_name.pop(old)
            entry["updated_at"] = _now()
            by_name[new] = entry
            return True

        self._mutate(mutate)

    def forget(self, kind: str, scope: str | os.PathLike[str], name: str) -> bool:
        """Drop an alias. Returns whether it existed. The native session
        itself is untouched — only the name is forgotten."""
        kind, scope_key, name = _check_kind(kind), normalize_scope(scope), _check_label(name)
        removed = False

        def mutate(sessions: dict[str, Any]) -> bool:
            nonlocal removed
            by_scope = sessions.get(kind, {})
            by_name = by_scope.get(scope_key, {})
            if name not in by_name:
                return False
            del by_name[name]
            if not by_name:
                del by_scope[scope_key]
            removed = True
            return True

        self._mutate(mutate)
        return removed

    def entries(self, kind: str | None = None, scope: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
        """Every entry as a flat dict, optionally filtered, sorted by key.

        Each dict carries ``kind``, ``scope``, ``name``, ``native_id``,
        ``model``, ``effort``, ``created_at`` and ``updated_at``.
        """
        if kind is not None:
            _check_kind(kind)
        scope_key = normalize_scope(scope) if scope is not None else None
        rows: list[dict[str, Any]] = []
        for k, by_scope in self._load()["sessions"].items():
            if kind is not None and k != kind:
                continue
            for s, by_name in by_scope.items():
                if scope_key is not None and s != scope_key:
                    continue
                for n, entry in by_name.items():
                    rows.append(
                        {
                            "kind": k,
                            "scope": s,
                            "name": n,
                            "native_id": entry["native_id"],
                            "model": entry.get("model"),
                            "effort": entry.get("effort"),
                            "created_at": entry.get("created_at"),
                            "updated_at": entry.get("updated_at"),
                        }
                    )
        rows.sort(key=lambda r: (r["kind"], r["scope"], r["name"]))
        return rows

    # -- storage -----------------------------------------------------------

    def _read(self, path: Path) -> dict[str, Any]:
        """Parse the file; ``{"sessions": {}}`` if absent; raise if corrupt."""
        text: str | None = None
        for attempt in range(_IO_RETRIES):
            try:
                text = path.read_text(encoding="utf-8")
                break
            except FileNotFoundError:
                return {"version": _FORMAT_VERSION, "sessions": {}}
            except PermissionError:
                if attempt == _IO_RETRIES - 1:
                    raise
                time.sleep(_IO_RETRY_SLEEP)
        assert text is not None
        if not text.strip():
            # An empty file carries nothing worth quarantining.
            return {"version": _FORMAT_VERSION, "sessions": {}}
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise _CorruptRegistry(str(exc)) from exc
        if not _valid_shape(data):
            raise _CorruptRegistry("unexpected document shape")
        return dict(data)

    def _quarantine(self, path: Path, reason: str) -> None:
        target = path.with_name(path.name + ".corrupt")
        if target.exists():
            target = path.with_name(f"{path.name}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}")
        os.replace(path, target)
        warnings.warn(
            f"SessionRegistry: {path} is not a valid registry ({reason}); moved it to {target} and starting empty.",
            RuntimeWarning,
            stacklevel=4,
        )

    def _load(self) -> dict[str, Any]:
        """A consistent snapshot. Lock-free unless the file is corrupt."""
        path = self.path
        try:
            return self._read(path)
        except _CorruptRegistry:
            pass
        with self._exclusive(path):
            return self._read_or_quarantine(path)

    def _read_or_quarantine(self, path: Path) -> dict[str, Any]:
        # Caller holds the lock. Re-read: another process may have repaired it.
        try:
            return self._read(path)
        except _CorruptRegistry as exc:
            self._quarantine(path, str(exc))
            return {"version": _FORMAT_VERSION, "sessions": {}}

    def _mutate(self, fn: Any) -> None:
        """Read-modify-write under both locks. ``fn(sessions)`` returns
        whether it changed anything (so a no-op never rewrites the file)."""
        path = self.path
        with self._exclusive(path):
            data = self._read_or_quarantine(path)
            if fn(data["sessions"]):
                data["version"] = _FORMAT_VERSION
                self._write(path, data)

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            for attempt in range(_IO_RETRIES):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == _IO_RETRIES - 1:
                        raise
                    time.sleep(_IO_RETRY_SLEEP)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink()

    @contextlib.contextmanager
    def _exclusive(self, path: Path) -> Any:
        """Cross-thread and cross-process exclusion for one registry file.

        The in-process lock is taken first so threads queue cheaply; the
        ``<file>.lock`` file (created with ``O_EXCL``, which is atomic on every
        platform) excludes other processes. A lock older than
        ``_LOCK_STALE_SECONDS`` is taken to belong to a crashed process.
        """
        with self._tlock:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.with_name(path.name + ".lock")
            deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    break
                except (FileExistsError, PermissionError):
                    # PermissionError: Windows reports a lock file that is
                    # mid-deletion by its owner this way.
                    with contextlib.suppress(OSError):
                        if time.time() - lock_path.stat().st_mtime > _LOCK_STALE_SECONDS:
                            lock_path.unlink()
                            continue
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"could not lock session registry {path} within {_LOCK_TIMEOUT_SECONDS}s"
                        ) from None
                    time.sleep(0.005 + random.random() * 0.02)
            try:
                with contextlib.suppress(OSError):
                    os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                fd = -1
                yield
            finally:
                if fd != -1:
                    os.close(fd)
                with contextlib.suppress(OSError):
                    lock_path.unlink()


# -- module default ---------------------------------------------------------

_default_lock = threading.Lock()
_default_registry: SessionRegistry | None = None


def default_session_registry() -> SessionRegistry:
    """The process-wide registry engines use when given no ``session_registry``.

    Created on first use with the default location (``LAZYBRIDGE_SESSIONS_FILE``
    or ``~/.lazybridge/sessions.json``); replace it with
    :func:`set_default_session_registry`.
    """
    global _default_registry
    with _default_lock:
        if _default_registry is None:
            _default_registry = SessionRegistry()
        return _default_registry


def set_default_session_registry(registry: SessionRegistry | None) -> SessionRegistry | None:
    """Install ``registry`` as the default (``None`` restores lazy creation).

    Returns the previous override, so a test can put it back.
    """
    global _default_registry
    with _default_lock:
        previous, _default_registry = _default_registry, registry
    return previous


# -- engine glue ------------------------------------------------------------


class AliasBinding:
    """One engine's link to one alias (internal; both engines share it).

    Keeps the engine code down to three calls: ``known_id()`` at construction
    and at the start of a first run, and ``bind()`` wherever the engine absorbs
    a native id. ``bind()`` never raises: a registry that cannot be written
    (read-only disk, lock timeout) must not fail a turn that already ran, so it
    warns instead.
    """

    def __init__(
        self,
        kind: str,
        alias: str,
        scope: str | os.PathLike[str] | None,
        registry: SessionRegistry | None,
    ) -> None:
        self.kind = _check_kind(kind)
        self.alias = _check_label(alias)
        self.scope = normalize_scope(scope if scope is not None else os.getcwd())
        self.registry = registry if registry is not None else default_session_registry()
        self._last: tuple[str, str | None, str | None] | None = None

    def known_id(self) -> str | None:
        try:
            return self.registry.resolve(self.kind, self.scope, self.alias)
        except Exception as exc:  # an unreadable registry means "unknown", loudly
            warnings.warn(f"session alias {self.alias!r}: could not read the registry: {exc}", stacklevel=3)
            return None

    def begin_run(self) -> None:
        """Forget what was last written so this run refreshes ``updated_at``."""
        self._last = None

    def bind(self, native_id: str, *, model: str | None, effort: str | None) -> None:
        key = (native_id, model, effort)
        if key == self._last:
            return  # already recorded during this run (absorb runs on several paths)
        try:
            self.registry.bind(self.kind, self.scope, self.alias, native_id, model=model, effort=effort)
        except Exception as exc:
            warnings.warn(
                f"session alias {self.alias!r}: could not record native id {native_id!r}: {exc}",
                stacklevel=3,
            )
            return
        self._last = key
