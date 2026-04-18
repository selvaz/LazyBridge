"""StorePanel — GUI for ``LazyStore`` (in-memory or SQLite-backed).

Inspect tab lists every key with a short value preview and the writing
agent's id.  Test tab supports read-by-key, write, and delete operations;
``read all`` returns a JSON snapshot.
"""

from __future__ import annotations

import json
from typing import Any

from lazybridge.gui._panel import Panel

_PREVIEW_LIMIT = 160

#: Sentinel agent_id used for writes/deletes performed through the GUI.
#: Double-underscore brackets prevent collision with a real agent name.
GUI_AGENT_ID = "__gui_panel__"


class StorePanel(Panel):
    """Panel for a :class:`~lazybridge.lazy_store.LazyStore` instance."""

    kind = "store"

    def __init__(self, store: Any, *, label: str | None = None) -> None:
        self._store = store
        self._label_override = label

    @property
    def id(self) -> str:
        # LazyStore has no id of its own; use the object identity.
        return f"store-{id(self._store):x}"

    @property
    def label(self) -> str:
        if self._label_override:
            return self._label_override
        try:
            count = len(self._store.keys())
        except Exception:  # pragma: no cover - defensive
            count = 0
        return f"store · {count} key(s)"

    # ------------------------------------------------------------------

    def _preview(self, value: Any) -> str:
        try:
            s = value if isinstance(value, str) else json.dumps(value, default=repr)
        except Exception:
            s = repr(value)
        return s if len(s) <= _PREVIEW_LIMIT else s[:_PREVIEW_LIMIT] + "…"

    def _entry_dicts(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        try:
            raw_entries = self._store.entries()
        except Exception:
            # Fallback to keys/read_all if entries() not available.
            try:
                data = self._store.read_all()
            except Exception:
                return []
            return [
                {"key": k, "preview": self._preview(v), "agent_id": None, "written_at": None}
                for k, v in data.items()
            ]
        for e in raw_entries:
            entries.append(
                {
                    "key": e.key,
                    "preview": self._preview(e.value),
                    "agent_id": getattr(e, "agent_id", None),
                    "written_at": _format_written_at(getattr(e, "written_at", None)),
                }
            )
        return entries

    def render_state(self) -> dict[str, Any]:
        backend = getattr(self._store, "_backend", None)
        backend_kind = type(backend).__name__ if backend is not None else type(self._store).__name__
        return {
            "backend": backend_kind,
            "entries": self._entry_dicts(),
            "key_count": len(self._store.keys()),
        }

    # ------------------------------------------------------------------

    def handle_action(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "read":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("'key' is required")
            value = self._store.read(key)
            return {"key": key, "value": _jsonable(value)}

        if action == "write":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("'key' is required")
            raw = args.get("value", "")
            as_json = bool(args.get("as_json", False))
            if as_json:
                try:
                    value = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError as exc:
                    raise ValueError(f"value is not valid JSON: {exc}") from exc
            else:
                value = raw
            self._store.write(key, value, agent_id=GUI_AGENT_ID)
            return {"ok": True, "key": key, "preview": self._preview(value)}

        if action == "delete":
            key = args.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError("'key' is required")
            if key not in self._store:
                return {"ok": False, "reason": "not found"}
            self._store.delete(key)
            return {"ok": True, "key": key}

        if action == "read_all":
            return {"all": _jsonable(self._store.read_all())}

        return super().handle_action(action, args)


def _format_written_at(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    return repr(value)
