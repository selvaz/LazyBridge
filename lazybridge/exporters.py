"""Pluggable event exporters for LazyBridge tracking.

Exporters receive every event emitted by ``EventLog.log()`` and forward
them to external systems (Langfuse, OpenTelemetry, custom HTTP, etc.).

The core ``EventLog`` is untouched — exporters are an additional output
channel, not a replacement.

Quick start::

    from lazybridge import LazySession, CallbackExporter

    events = []
    sess = LazySession(exporters=[CallbackExporter(events.append)])
    # ... run agents ...
    print(events)  # list of event dicts

Filtering::

    from lazybridge.exporters import FilteredExporter, CallbackExporter

    tool_only = FilteredExporter(
        CallbackExporter(print),
        event_types={"tool_call", "tool_result"},
    )
    sess = LazySession(exporters=[tool_only])
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — any object with .export(event: dict) qualifies
# ---------------------------------------------------------------------------

@runtime_checkable
class EventExporter(Protocol):
    """Protocol for event exporters.

    Implement ``export(event)`` to receive events from ``EventLog``.
    Events are dicts with keys: timestamp, session_id, agent_id,
    agent_name, event_type, data.
    """

    def export(self, event: dict[str, Any]) -> None:
        """Receive a single event. Must not raise."""
        ...


# ---------------------------------------------------------------------------
# CallbackExporter — wraps a simple function
# ---------------------------------------------------------------------------

class CallbackExporter:
    """Wraps a callable as an EventExporter.

    Usage::

        exporter = CallbackExporter(lambda e: print(e["event_type"]))
        sess = LazySession(exporters=[exporter])
    """

    __slots__ = ("_fn",)

    def __init__(self, fn: Any) -> None:
        if not callable(fn):
            raise TypeError(f"Expected callable, got {type(fn).__name__}")
        self._fn = fn

    def export(self, event: dict[str, Any]) -> None:
        self._fn(event)

    def __repr__(self) -> str:
        return f"CallbackExporter({self._fn!r})"


# ---------------------------------------------------------------------------
# FilteredExporter — only forwards specified event types
# ---------------------------------------------------------------------------

class FilteredExporter:
    """Wraps another exporter, forwarding only specified event types.

    Usage::

        inner = CallbackExporter(my_handler)
        exporter = FilteredExporter(inner, event_types={"tool_call", "model_response"})
    """

    __slots__ = ("_inner", "_types")

    def __init__(self, inner: EventExporter, *, event_types: set[str]) -> None:
        self._inner = inner
        self._types = frozenset(event_types)

    def export(self, event: dict[str, Any]) -> None:
        if event.get("event_type") in self._types:
            self._inner.export(event)

    def __repr__(self) -> str:
        return f"FilteredExporter({self._inner!r}, types={set(self._types)})"


# ---------------------------------------------------------------------------
# JsonFileExporter — append events as JSON lines to a file
# ---------------------------------------------------------------------------

class JsonFileExporter:
    """Appends events as JSON lines to a file.

    Useful for offline analysis, debugging, or feeding into other tools.
    The file handle is kept open for performance and flushed after each
    write to ensure durability.

    Usage::

        exporter = JsonFileExporter("events.jsonl")
        sess = LazySession(exporters=[exporter])
        # ... run agents ...
        exporter.close()  # optional — flushes and closes
    """

    __slots__ = ("_path", "_fh", "_lock")

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def export(self, event: dict[str, Any]) -> None:
        import json
        try:
            line = json.dumps(event, default=str)
            with self._lock:
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception as exc:
            _logger.debug("JsonFileExporter write failed: %s", exc)

    def close(self) -> None:
        """Flush and close the file handle."""
        try:
            with self._lock:
                self._fh.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"JsonFileExporter({self._path!r})"
