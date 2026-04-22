"""Exporters — pluggable event sinks for Session observability."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EventExporter(Protocol):
    """Protocol satisfied by all exporter classes."""

    def export(self, event: dict[str, Any]) -> None: ...


class CallbackExporter:
    """Forward every event to a user-supplied callable."""

    def __init__(self, fn: Callable[[dict[str, Any]], None]) -> None:
        self._fn = fn

    def export(self, event: dict[str, Any]) -> None:
        self._fn(event)


class FilteredExporter:
    """Forward only events whose type is in ``event_types`` to ``inner``."""

    def __init__(self, inner: Any, *, event_types: set[str]) -> None:
        self._inner = inner
        self._types = event_types

    def export(self, event: dict[str, Any]) -> None:
        if event.get("event_type") in self._types:
            self._inner.export(event)


class JsonFileExporter:
    """Append each event as a JSON line to ``path``.

    F7: keeps the file handle open across calls instead of opening and closing
    it on every event.  Under a typical agent run with 50-200 events the
    original per-call open/fwrite/close caused O(n) filesystem syscalls.
    ``close()`` is called automatically by ``Session.close()`` when it
    iterates its exporter list.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def export(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle. Idempotent."""
        try:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover
        self.close()


class StructuredLogExporter:
    """Emit each event via Python's ``logging`` module."""

    def __init__(self, logger_name: str = "lazybridge") -> None:
        self._log = logging.getLogger(logger_name)

    def export(self, event: dict[str, Any]) -> None:
        self._log.info(json.dumps(event, default=str))


class ConsoleExporter:
    """Pretty-print events to stdout for human inspection.

    Output format (one line per event)::

        [agent_name] event_type  key=value  key=value

    Installed automatically by ``Session(console=True)`` and
    ``Agent(verbose=True)``; can also be added manually via
    ``Session(exporters=[ConsoleExporter()])``.
    """

    _NOISY_KEYS = {"session_id", "event_type", "run_id"}

    def __init__(self, *, stream: Any = None) -> None:
        import sys

        self._stream = stream or sys.stdout

    def export(self, event: dict[str, Any]) -> None:
        etype = event.get("event_type", "event")
        agent = event.get("agent_name") or event.get("name") or ""
        parts: list[str] = []
        for k, v in event.items():
            if k in self._NOISY_KEYS or k == "agent_name":
                continue
            if v is None or v == "":
                continue
            s = str(v)
            if len(s) > 120:
                s = s[:117] + "..."
            parts.append(f"{k}={s}")
        prefix = f"[{agent}] " if agent else ""
        line = f"{prefix}{etype}  " + "  ".join(parts)
        print(line, file=self._stream)


class OTelExporter:
    """Export events as OpenTelemetry spans (requires opentelemetry-sdk).

    Hierarchy: Agent span → tool/model child spans.
    Install: ``pip install lazybridge[otel]``

    Thread-safe: ``Session.emit`` can fan events out to this exporter
    from multiple worker threads, so the in-flight span registry is
    guarded by a lock.  Call :meth:`close` to flush any spans that are
    still open (e.g. when a run is cancelled before ``agent_finish``).
    """

    def __init__(self, endpoint: str | None = None, *, exporter: Any | None = None) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider = TracerProvider()
            if exporter:
                provider.add_span_processor(SimpleSpanProcessor(exporter))
            elif endpoint:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("lazybridge")
        except ImportError:
            raise ImportError("Install opentelemetry-sdk: pip install lazybridge[otel]") from None

        import threading

        self._spans: dict[str, Any] = {}
        self._lock = threading.Lock()

    def export(self, event: dict[str, Any]) -> None:
        from opentelemetry import trace

        et = event.get("event_type", "")
        run_id = event.get("run_id", "unknown")

        if et == "agent_start":
            span = self._tracer.start_span(f"agent:{event.get('agent_name', 'agent')}")
            with self._lock:
                self._spans[run_id] = span
        elif et in ("agent_finish", "agent_error"):
            with self._lock:
                span = self._spans.pop(run_id, None)
            if span:
                span.end()
        elif et in ("tool_call", "tool_result", "tool_error", "model_request", "model_response"):
            with self._lock:
                parent = self._spans.get(run_id)
            ctx = trace.set_span_in_context(parent) if parent else None
            child = self._tracer.start_span(et, context=ctx)
            for k, v in event.items():
                if k not in ("event_type", "run_id", "session_id"):
                    child.set_attribute(k, str(v))
            child.end()

    def close(self) -> None:
        """Flush any spans still open — e.g. after a cancelled run.

        Idempotent.  Without this, a run that crashes before emitting
        ``agent_finish`` leaves its span stuck in ``self._spans`` for
        the life of the process.
        """
        with self._lock:
            spans = list(self._spans.values())
            self._spans.clear()
        for span in spans:
            try:
                span.end()
            except Exception:
                pass
