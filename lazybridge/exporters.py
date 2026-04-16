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

    Or as a context manager::

        with JsonFileExporter("events.jsonl") as exporter:
            sess = LazySession(exporters=[exporter])
            # ... run agents ...
    """

    __slots__ = ("_fh", "_lock", "_path")

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

    def __enter__(self) -> JsonFileExporter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"JsonFileExporter({self._path!r})"


# ---------------------------------------------------------------------------
# StructuredLogExporter — emit events as structured JSON log lines
# ---------------------------------------------------------------------------


class StructuredLogExporter:
    """Emits events as structured JSON log lines via Python's logging module.

    Usage::

        import logging
        logging.basicConfig(level=logging.INFO)
        exporter = StructuredLogExporter()
        sess = LazySession(exporters=[exporter])
    """

    __slots__ = ("_level", "_logger")

    def __init__(self, logger_name: str = "lazybridge.events", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    def export(self, event: dict[str, Any]) -> None:
        import json

        try:
            self._logger.log(self._level, json.dumps(event, default=str))
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"StructuredLogExporter({self._logger.name!r})"


# ---------------------------------------------------------------------------
# OTelExporter — map LazyBridge events to OpenTelemetry spans
# ---------------------------------------------------------------------------


class OTelExporter:
    """Maps LazyBridge events to OpenTelemetry spans.

    Requires ``pip install lazybridge[otel]`` (opentelemetry-api + opentelemetry-sdk).

    Usage::

        from lazybridge.exporters import OTelExporter

        exporter = OTelExporter(service_name="my-pipeline")
        sess = LazySession(exporters=[exporter])

    Span mapping:
    - ``agent_start`` → opens a new span named after the agent
    - ``agent_finish`` → closes the span with token/cost attributes
    - ``tool_call`` → child span under the agent span
    - ``tool_result`` → ends the tool span
    - ``tool_error`` → records error on the tool span
    - ``model_request`` / ``model_response`` → child span for LLM calls
    """

    __slots__ = ("_agent_spans", "_lock", "_model_spans", "_tool_spans", "_tracer")

    def __init__(self, service_name: str = "lazybridge", tracer_name: str = "lazybridge") -> None:
        try:
            from opentelemetry import trace
        except ImportError:
            raise ImportError("OpenTelemetry not installed. Run: pip install lazybridge[otel]") from None

        self._tracer = trace.get_tracer(tracer_name, schema_url=f"https://lazybridge.dev/schema/{service_name}")
        self._agent_spans: dict[str, Any] = {}
        self._tool_spans: dict[str, Any] = {}
        self._model_spans: dict[str, Any] = {}
        self._lock = threading.Lock()

    def export(self, event: dict[str, Any]) -> None:
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode

        event_type = event.get("event_type", "")
        agent_id = event.get("agent_id", "")
        agent_name = event.get("agent_name", "unknown")
        data = event.get("data", {}) if isinstance(event.get("data"), dict) else {}

        with self._lock:
            if event_type == "agent_start":
                parent = self._agent_spans.get(agent_id)
                ctx = trace.set_span_in_context(parent) if parent else None
                span = self._tracer.start_span(f"agent:{agent_name}", context=ctx)
                span.set_attribute("lazybridge.agent.name", agent_name)
                span.set_attribute("lazybridge.agent.id", agent_id)
                if "task" in data:
                    span.set_attribute("lazybridge.task", str(data["task"])[:200])
                self._agent_spans[agent_id] = span

            elif event_type == "agent_finish":
                span = self._agent_spans.pop(agent_id, None)
                if span:
                    for key in ("stop_reason", "n_steps"):
                        if key in data:
                            span.set_attribute(f"lazybridge.{key}", str(data[key]))
                    span.end()

            elif event_type == "model_request":
                parent = self._agent_spans.get(agent_id)
                ctx = trace.set_span_in_context(parent) if parent else None
                span = self._tracer.start_span(f"llm:{data.get('model', 'unknown')}", context=ctx)
                span.set_attribute("llm.model", str(data.get("model", "")))
                self._model_spans[agent_id] = span

            elif event_type == "model_response":
                span = self._model_spans.pop(agent_id, None)
                if span:
                    for key, attr in [
                        ("input_tokens", "llm.input_tokens"),
                        ("output_tokens", "llm.output_tokens"),
                        ("cost_usd", "llm.cost_usd"),
                    ]:
                        if key in data:
                            span.set_attribute(attr, data[key])
                    span.end()

            elif event_type == "tool_call":
                parent = self._agent_spans.get(agent_id)
                ctx = trace.set_span_in_context(parent) if parent else None
                tool_name = str(data.get("name", "unknown"))
                span = self._tracer.start_span(f"tool:{tool_name}", context=ctx)
                span.set_attribute("lazybridge.tool.name", tool_name)
                self._tool_spans[f"{agent_id}:{tool_name}"] = span

            elif event_type == "tool_result":
                tool_name = str(data.get("name", "unknown"))
                key = f"{agent_id}:{tool_name}"
                span = self._tool_spans.pop(key, None)
                if span:
                    span.end()

            elif event_type == "tool_error":
                tool_name = str(data.get("name", "unknown"))
                key = f"{agent_id}:{tool_name}"
                span = self._tool_spans.pop(key, None)
                if span:
                    span.set_status(StatusCode.ERROR, str(data.get("error", "")))
                    span.end()

    def __repr__(self) -> str:
        return "OTelExporter()"
