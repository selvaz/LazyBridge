## signature
Session(
    *,
    db: str | None = None,            # None = in-memory SQLite (per session_id)
    exporters: list[EventExporter] = None,
    redact: Callable[[dict], dict] | None = None,
    redact_on_error: Literal["fallback", "strict"] = "strict",
    console: bool = False,            # add a ConsoleExporter
    # Batched-writer (opt-in) — submit events from the hot path,
    # let a background thread INSERT in batches.
    batched: bool = False,
    batch_size: int = 100,
    batch_interval: float = 1.0,
    max_queue_size: int = 10_000,
    on_full: Literal["drop", "block", "hybrid"] = "hybrid",
    critical_events: frozenset[str] | None = None,  # None = framework default
) -> Session

session.emit(event_type: EventType, payload: dict, *, run_id: str = None) -> None
session.add_exporter(exporter: EventExporter) -> None
session.remove_exporter(exporter: EventExporter) -> None
session.flush(timeout: float = 5.0) -> None        # drain batched writer
session.close() -> None                            # release SQLite + flush exporters
session.usage_summary() -> {"total": {...}, "by_agent": {...}, "by_run": {...}}

session.events: EventLog          # session.events.query(...) for raw rows
session.graph:  GraphSchema       # auto-populated when Agents register

EventType (StrEnum):
  AGENT_START  AGENT_FINISH
  LOOP_STEP
  MODEL_REQUEST  MODEL_RESPONSE
  TOOL_CALL  TOOL_RESULT  TOOL_ERROR
  HIL_DECISION

# Agent("model", verbose=True) creates a private Session(console=True).

## rules
- Every engine emits events with the same enum. Hand an Agent a
  ``session=`` and you get a full per-run trace.
- ``redact`` runs on every payload. ``redact_on_error="strict"``
  (default) drops the event when the redactor raises or returns a
  non-dict — fail-closed. ``"fallback"`` warns once, then records the
  unredacted payload.
- Nested Agents (Agent A has Agent B as a tool) inherit the outer
  session. All events flow to one EventLog so ``usage_summary()`` can
  aggregate cost across the whole tree.
- ``batched=True`` makes ``emit`` non-blocking. Saturation policy
  (``on_full=``):
    * ``"hybrid"`` — block on critical events (``AGENT_*`` /
      ``TOOL_*`` / ``HIL_DECISION``), drop ``LOOP_STEP`` /
      ``MODEL_REQUEST`` / ``MODEL_RESPONSE``. **Default.**
    * ``"block"`` — back-pressure unconditionally.
    * ``"drop"``  — drop on saturation with a doubling-interval warning.
- ``critical_events=`` overrides the hybrid set. Pass an empty set to
  get drop-all-on-saturation behaviour while keeping hybrid as the policy.
- Exporters fire in registration order. A failing exporter warns once
  per instance; subsequent failures from the same exporter are
  suppressed.

## narrative
`Session` is the observability container: an SQLite-backed `EventLog`
plus a list of exporters (Console / JSON-file / OTel / your own).
Agents emit a small, fixed event schema; the session aggregates,
persists, and fans out.

For dev, `Agent(verbose=True)` (or `Session(console=True)`) is enough —
it prints one line per event. For production, pair `JsonFileExporter`
or `OTelExporter` with `batched=True` so the hot path doesn't block on
disk / network.

The `"hybrid"` back-pressure default exists because audit-relevant
events (an agent finishing, a tool failing, a human decision) must
never silently disappear under load, while cheap telemetry can. If you
need every single event no matter the cost, use `on_full="block"`.

## example
```python
from lazybridge import Agent, Session, ConsoleExporter, JsonFileExporter
from lazybridge.session import EventType
from lazybridge.ext.otel import OTelExporter

# Dev — stdout tracing with one flag.
sess = Session(console=True)
Agent("claude-opus-4-7", name="chat", session=sess)("hello")

# Prod — multi-sink observability with batched writer.
sess = Session(
    db="events.sqlite",
    batched=True,
    on_full="hybrid",                 # default; explicit for clarity
    exporters=[
        JsonFileExporter("events.jsonl"),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
    redact=lambda p: {**p, "task": _mask_pii(p.get("task", ""))},
)
agents = [researcher, writer]
pipeline = Agent.chain(*agents, session=sess)
pipeline("summarise AI trends")

# Cost / token roll-up across the whole tree.
print(sess.usage_summary()["total"]["cost_usd"])

# Drain the writer before reading the log (or use a context manager).
sess.flush()
print(sess.events.query(event_type=EventType.TOOL_ERROR))

# Topology for a UI.
print(sess.graph.to_json())
```

## pitfalls
- ``Session(db=":memory:")`` behaves like ``Session()`` (in-memory).
  Use a filename to persist.
- Exporter failures warn ONCE per exporter instance. If a third
  failure mode shows up, you'll only see the first — wrap in
  ``CallbackExporter(lambda e: print(e))`` while debugging.
- ``Agent(verbose=True)`` creates a **new** Session for that agent.
  If you also pass ``session=another``, ``verbose`` is ignored.
- With ``batched=True`` reads via ``session.events.query()`` may be
  stale until ``session.flush()`` (or ``close()``) drains the writer.
- ``on_full="drop"`` was the pre-1.0.x default and is still available;
  the 1.0.x release flipped the default to ``"hybrid"`` so an
  ``AGENT_FINISH`` or ``TOOL_ERROR`` is never silently lost.

## see-also
- [Exporters](exporters.md) — the sinks that consume `Session` events.
- [Memory](memory.md) — separate concept (conversation context).
- [GraphSchema](graph-schema.md) — the agent topology view.
