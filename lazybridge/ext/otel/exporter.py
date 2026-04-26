"""``OTelExporter`` — emit Session events as OpenTelemetry spans.

Moved from :mod:`lazybridge.exporters` in lazybridge 1.0.0 as part of the
core-vs-ext split (see ``docs/guides/core-vs-ext.md``). Backwards-compat
re-exports remain at the old import paths until lazybridge 1.2.
"""

from __future__ import annotations

from typing import Any


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
