"""``OTelExporter`` — emit Session events as OpenTelemetry GenAI spans.

Conforms to the OpenTelemetry Semantic Conventions for Generative AI
(``gen_ai.*`` attributes) so dashboards built for the standard
(Datadog GenAI, Honeycomb GenAI, Grafana Tempo) render LazyBridge
traces without translation.

Span hierarchy:

    invoke_agent  <agent_name>          (root for one Agent.run)
      ├─ chat       <model>             (one per LLM round-trip)
      └─ execute_tool <tool_name>       (one per tool invocation)
            └─ invoke_agent <inner>     (when the tool is itself an Agent)

Tool spans run as children of the agent span and are correctly closed
on TOOL_RESULT / TOOL_ERROR (correlated by ``tool_use_id``).  Cross-
agent parenting works automatically: the inner agent's events are
emitted on the same asyncio context as the outer tool span, so OTel's
contextvars-based propagation makes the inner span a child of the
outer one without any explicit run-id chaining.

Moved from :mod:`lazybridge.exporters` in lazybridge 1.0.0 as part of
the core-vs-ext split (see ``docs/guides/core-vs-ext.md``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# OpenTelemetry GenAI Semantic Conventions — attribute names
# ---------------------------------------------------------------------------

# Frozen here so a future SDK rev that ships them as constants can swap
# this in without touching the export logic.
_GA_SYSTEM = "gen_ai.system"
_GA_OPERATION = "gen_ai.operation.name"
_GA_REQ_MODEL = "gen_ai.request.model"
_GA_RESP_MODEL = "gen_ai.response.model"
_GA_USAGE_IN = "gen_ai.usage.input_tokens"
_GA_USAGE_OUT = "gen_ai.usage.output_tokens"
_GA_FINISH_REASONS = "gen_ai.response.finish_reasons"
_GA_TOOL_NAME = "gen_ai.tool.name"
_GA_TOOL_CALL_ID = "gen_ai.tool.call.id"
_GA_AGENT_NAME = "gen_ai.agent.name"

# LazyBridge-specific (no GenAI equivalent yet).  Prefixed under
# ``lazybridge.*`` so an operator can filter on them deterministically
# without mistaking them for a future GenAI rename.
_LB_RUN_ID = "lazybridge.run_id"
_LB_COST_USD = "lazybridge.cost_usd"
_LB_TURN = "lazybridge.turn"
_LB_BRANCH_ID = "lazybridge.branch_id"


@dataclass
class _SpanEntry:
    """One open span plus the OTel context-detach token that retires it."""

    span: Any
    detach_token: Any


class OTelExporter:
    """Export events as OpenTelemetry spans (requires opentelemetry-sdk).

    Emits ``gen_ai.*`` attributes per the OpenTelemetry Semantic
    Conventions for GenAI, with proper parent-child span hierarchy.

    Install: ``pip install lazybridge[otel]``

    Thread-safe: ``Session.emit`` can fan events out to this exporter
    from multiple worker threads, so the in-flight span registry is
    guarded by a lock.  Call :meth:`close` to flush any spans that are
    still open (e.g. when a run is cancelled before ``agent_finish``).

    The exporter sets each span as the *current* OTel context span
    while it is open, so nested agents (Agent-as-tool) automatically
    inherit the outer tool span as their parent without any explicit
    correlation id — OTel's contextvars-based propagation does the
    work.
    """

    def __init__(self, *, endpoint: str | None = None, exporter: Any | None = None) -> None:
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
            # Keep a per-instance tracer rooted in *our* provider so
            # multiple ``OTelExporter`` instances in one process (test
            # suites, multi-tenant agents) don't fight over the global
            # provider.  Cross-agent parenting still works through the
            # OTel context (contextvars), which is provider-agnostic.
            self._provider = provider
            self._tracer = provider.get_tracer("lazybridge")
            # Best-effort install as the global provider so ad-hoc OTel
            # users (their own ``trace.get_tracer(...)`` calls) see
            # spans without an explicit handle to ours.  No-op if
            # another component already set one.
            try:
                trace.set_tracer_provider(provider)
            except Exception:
                pass
        except ImportError:
            raise ImportError("Install opentelemetry-sdk: pip install lazybridge[otel]") from None

        # Outer key: run_id.  Inner key: ``"agent"``, ``"model"``, or
        # ``f"tool:{tool_use_id_or_name}"``.  Guarded by ``self._lock``.
        self._spans: dict[str, dict[str, _SpanEntry]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Span lifecycle helpers
    # ------------------------------------------------------------------

    def _start_span(self, run_id: str, key: str, name: str, *, parent_key: str | None = None) -> Any:
        """Open a span, register it under ``(run_id, key)``, and attach
        it to the OTel context so child spans pick it up as parent.

        If ``parent_key`` is given, the new span is parented to that
        registered entry.  Otherwise OTel's current context is used —
        which means an outer tool span (or an outer agent span emitted
        by a parent Agent on the same asyncio context) is picked up
        automatically.
        """
        from opentelemetry import context as ot_context
        from opentelemetry import trace

        ctx = None
        with self._lock:
            run_spans = self._spans.get(run_id)
            if parent_key and run_spans and parent_key in run_spans:
                ctx = trace.set_span_in_context(run_spans[parent_key].span)
        span = self._tracer.start_span(name, context=ctx)
        # Attach span to the current context so genuinely nested
        # operations (an inner Agent.run() invoked from a tool) inherit
        # this span as their parent without needing the run_id linkage.
        token = ot_context.attach(trace.set_span_in_context(span))
        with self._lock:
            self._spans.setdefault(run_id, {})[key] = _SpanEntry(span=span, detach_token=token)
        return span

    def _end_span(self, run_id: str, key: str, *, error: str | None = None) -> Any | None:
        """Close the span registered at ``(run_id, key)`` and detach it
        from the OTel context.  Returns the closed span (or ``None``
        when none was open)."""
        from opentelemetry import context as ot_context
        from opentelemetry import trace

        with self._lock:
            run_spans = self._spans.get(run_id)
            if not run_spans:
                return None
            entry = run_spans.pop(key, None)
            if not run_spans:
                self._spans.pop(run_id, None)
        if entry is None:
            return None
        if error is not None:
            try:
                entry.span.set_status(trace.Status(trace.StatusCode.ERROR, error))
            except Exception:
                pass
        try:
            ot_context.detach(entry.detach_token)
        except Exception:
            pass
        try:
            entry.span.end()
        except Exception:
            pass
        return entry.span

    # ------------------------------------------------------------------
    # Event router
    # ------------------------------------------------------------------

    def export(self, event: dict[str, Any]) -> None:
        et = event.get("event_type", "")
        run_id = event.get("run_id") or "unknown"

        if et == "agent_start":
            self._on_agent_start(run_id, event)
        elif et in ("agent_finish", "agent_error"):
            self._on_agent_end(run_id, event)
        elif et == "model_request":
            self._on_model_request(run_id, event)
        elif et == "model_response":
            self._on_model_response(run_id, event)
        elif et == "tool_call":
            self._on_tool_call(run_id, event)
        elif et in ("tool_result", "tool_error"):
            self._on_tool_end(run_id, event, errored=(et == "tool_error"))
        elif et == "loop_step":
            # Pure progress signal — turn count, message count.  Emit as
            # a child event on the agent span without opening a span.
            self._maybe_add_event(run_id, "loop_step", event)
        elif et == "hil_decision":
            self._maybe_add_event(run_id, "hil_decision", event)

    # ------------------------------------------------------------------
    # Per-event-type handlers
    # ------------------------------------------------------------------

    def _on_agent_start(self, run_id: str, event: dict[str, Any]) -> None:
        agent_name = event.get("agent_name") or "agent"
        span = self._start_span(run_id, "agent", f"invoke_agent {agent_name}")
        self._set_attr(span, _GA_OPERATION, "invoke_agent")
        self._set_attr(span, _GA_AGENT_NAME, agent_name)
        self._set_attr(span, _LB_RUN_ID, run_id)
        if "task" in event:
            self._set_attr(span, "gen_ai.prompt", _truncate(event["task"]))

    def _on_agent_end(self, run_id: str, event: dict[str, Any]) -> None:
        # Update the agent span with whatever final attributes are
        # available before closing it.  Other open children (model
        # mid-stream, tool that crashed without TOOL_ERROR) are closed
        # too so we don't leak open spans on a cancelled run.
        with self._lock:
            agent_entry = self._spans.get(run_id, {}).get("agent")
        if agent_entry is not None:
            if "payload" in event:
                self._set_attr(agent_entry.span, "gen_ai.completion", _truncate(event["payload"]))
            if "latency_ms" in event:
                self._set_attr(agent_entry.span, "lazybridge.latency_ms", event["latency_ms"])
            if event.get("cancelled"):
                self._set_attr(agent_entry.span, "lazybridge.cancelled", True)

        # Close any orphan children before the agent span itself.
        with self._lock:
            keys = [k for k in self._spans.get(run_id, {}) if k != "agent"]
        for k in keys:
            self._end_span(run_id, k, error="orphan: agent ended before child")

        err = event.get("error")
        self._end_span(run_id, "agent", error=str(err) if err else None)

    def _on_model_request(self, run_id: str, event: dict[str, Any]) -> None:
        model = event.get("model") or "model"
        span = self._start_span(run_id, "model", f"chat {model}", parent_key="agent")
        self._set_attr(span, _GA_OPERATION, "chat")
        if "provider" in event:
            self._set_attr(span, _GA_SYSTEM, _normalise_provider(event["provider"]))
        self._set_attr(span, _GA_REQ_MODEL, model)
        if "turn" in event:
            self._set_attr(span, _LB_TURN, event["turn"])

    def _on_model_response(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            entry = self._spans.get(run_id, {}).get("model")
        if entry is not None:
            span = entry.span
            if "input_tokens" in event:
                self._set_attr(span, _GA_USAGE_IN, event["input_tokens"])
            if "output_tokens" in event:
                self._set_attr(span, _GA_USAGE_OUT, event["output_tokens"])
            if "stop_reason" in event and event["stop_reason"] is not None:
                # GenAI spec: a list (in case of multi-choice responses).
                self._set_attr(span, _GA_FINISH_REASONS, [str(event["stop_reason"])])
            if "cost_usd" in event and event["cost_usd"] is not None:
                self._set_attr(span, _LB_COST_USD, event["cost_usd"])
            if "model" in event:
                self._set_attr(span, _GA_RESP_MODEL, event["model"])
        self._end_span(run_id, "model")

    def _on_tool_call(self, run_id: str, event: dict[str, Any]) -> None:
        # Correlate calls and results by tool_use_id when the engine
        # supplies one (LLMEngine does as of 1.0.x).  Fall back to the
        # tool name otherwise — the legacy single-in-flight path.
        tool_use_id = event.get("tool_use_id")
        tool_name = event.get("tool") or "tool"
        key = f"tool:{tool_use_id}" if tool_use_id else f"tool:{tool_name}"
        span = self._start_span(run_id, key, f"execute_tool {tool_name}", parent_key="agent")
        self._set_attr(span, _GA_OPERATION, "execute_tool")
        self._set_attr(span, _GA_TOOL_NAME, tool_name)
        if tool_use_id:
            self._set_attr(span, _GA_TOOL_CALL_ID, tool_use_id)
        if "branch_id" in event:
            self._set_attr(span, _LB_BRANCH_ID, event["branch_id"])
        if "arguments" in event:
            # Stringified for safety — arguments may contain non-JSON
            # types from upstream.
            self._set_attr(span, "gen_ai.tool.arguments", _truncate(event["arguments"]))

    def _on_tool_end(self, run_id: str, event: dict[str, Any], *, errored: bool) -> None:
        tool_use_id = event.get("tool_use_id")
        tool_name = event.get("tool") or "tool"
        key = f"tool:{tool_use_id}" if tool_use_id else f"tool:{tool_name}"
        with self._lock:
            entry = self._spans.get(run_id, {}).get(key)
        if entry is not None:
            if "result" in event:
                self._set_attr(entry.span, "gen_ai.tool.result", _truncate(event["result"]))
            if "error" in event:
                self._set_attr(entry.span, "exception.message", str(event["error"]))
            if "type" in event and errored:
                self._set_attr(entry.span, "exception.type", str(event["type"]))
        err_msg = str(event.get("error")) if errored else None
        self._end_span(run_id, key, error=err_msg)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_add_event(self, run_id: str, name: str, event: dict[str, Any]) -> None:
        """Record a non-span event on the active agent span, if any."""
        with self._lock:
            entry = self._spans.get(run_id, {}).get("agent")
        if entry is None:
            return
        attrs = {k: _stringify(v) for k, v in event.items() if k not in ("event_type", "run_id", "session_id")}
        try:
            entry.span.add_event(name, attributes=attrs)
        except Exception:
            pass

    @staticmethod
    def _set_attr(span: Any, key: str, value: Any) -> None:
        try:
            span.set_attribute(key, _stringify(value))
        except Exception:
            pass

    def close(self) -> None:
        """Flush any spans still open — e.g. after a cancelled run.

        Idempotent.  Without this, a run that crashes before emitting
        ``agent_finish`` leaves its spans stuck for the life of the
        process, and the OTel contextvars stay attached to nothing.
        """
        with self._lock:
            run_ids = list(self._spans.keys())
        for run_id in run_ids:
            with self._lock:
                keys = list(self._spans.get(run_id, {}).keys())
            for key in keys:
                self._end_span(run_id, key, error="exporter closed")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _truncate(value: Any, *, limit: int = 1024) -> str:
    """Stringify and truncate a payload value for span attributes."""
    s = _stringify(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _stringify(value: Any) -> Any:
    """OTel attribute values must be primitive — coerce dicts/lists to JSON.

    Lists of primitives pass through untouched (the SDK accepts them and
    GenAI conventions sometimes require them, e.g. ``finish_reasons``).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list) and all(isinstance(x, (bool, int, float, str)) for x in value):
        return value
    try:
        import json

        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _normalise_provider(provider_class_name: str) -> str:
    """Map a LazyBridge provider class name to its GenAI ``system`` value.

    The session payload sends ``provider`` as the provider class's
    ``__class__.__name__`` (e.g. ``"AnthropicProvider"``).  GenAI
    conventions expect a short well-known string (``"anthropic"``,
    ``"openai"``, …).
    """
    s = provider_class_name.lower()
    for known in ("anthropic", "openai", "google", "deepseek", "lmstudio", "litellm"):
        if known in s:
            return known
    return s
