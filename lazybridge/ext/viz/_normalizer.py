"""JSON-safe normalisation of event payloads.

Event payloads carry whatever the engine recorded — Pydantic models,
``datetime`` objects, ``ToolCall`` instances, raw bytes. The browser
only speaks JSON, so every value that crosses the SSE boundary has to
be turned into something ``json.dumps`` accepts. We keep the rules in
one place so the server, the exporter, and the replay controller all
agree on the wire shape.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

_MAX_STRING = 8192  # cap per string field — guards SSE frame size


def to_jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Best-effort coercion of ``value`` into a JSON-serialisable form.

    Falls back to ``repr()`` for anything we don't recognise rather than
    raising — losing fidelity is better than dropping the whole event.
    """
    if _depth > 8:
        return f"<truncated depth={_depth}>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "…"
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = bytes(value).decode("utf-8", errors="replace")
        except Exception:
            decoded = repr(bytes(value))
        return decoded if len(decoded) <= _MAX_STRING else decoded[:_MAX_STRING] + "…"
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, _depth=_depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth=_depth + 1) for v in value]
    # Pydantic v2
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return to_jsonable(dump(mode="json"), _depth=_depth + 1)
        except Exception:
            pass
    # dataclass-ish
    asdict = getattr(value, "__dict__", None)
    if isinstance(asdict, dict):
        try:
            return {k: to_jsonable(v, _depth=_depth + 1) for k, v in asdict.items() if not k.startswith("_")}
        except Exception:
            pass
    text = repr(value)
    return text if len(text) <= _MAX_STRING else text[:_MAX_STRING] + "…"


def normalise_event(event: dict[str, Any]) -> dict[str, Any]:
    """Make an in-process event dict ready for SSE transport.

    Returns a new dict; the input is not mutated.
    """
    return {str(k): to_jsonable(v) for k, v in event.items()}
