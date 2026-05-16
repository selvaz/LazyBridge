# Session

The observability container for an agent run — an event log plus a list
of exporters (console, JSON file, OpenTelemetry, your own). Every
engine emits the same event schema; nested agents inherit the caller's
session, so cost / token / latency rollup works transitively across
the whole tree.

## Signature

```python
from lazybridge import Session

Session(
    *,
    db=None,                       # None = in-memory SQLite (lost at close)
    exporters=None,                # list[EventExporter]; None = []
    # Redaction
    redact=...,                    # default: redact_secrets — sk-/ghp-/Bearer/JWT/etc. masked
    redact_on_error="strict",      # "strict" (drop on redact failure) | "fallback" (warn + record raw)
    unsafe_log_payloads=False,     # set True to disable default secret redaction; redact=None has same effect
    console=False,                 # convenience: append a ConsoleExporter
    # Batched-writer (opt-in) — emit() becomes non-blocking
    batched=False,
    batch_size=100,
    batch_interval=1.0,
    max_queue_size=10_000,
    on_full="hybrid",              # "hybrid" (default) | "block" | "drop"
    critical_events=None,          # frozenset[str] — overrides hybrid set
)

# Methods
session.emit(event_type, payload, *, run_id=None)
session.add_exporter(exporter)
session.remove_exporter(exporter)
session.flush(timeout=5.0)         # drain the batched writer
session.close()                    # flush + release SQLite
session.usage_summary()            # {"total": {...}, "by_agent": {...}, "by_run": {...}}

# Live members
session.events                     # EventLog — session.events.query(...) for raw rows
session.graph                      # GraphSchema — agent topology, auto-populated
```

### EventType (StrEnum)

| Member | Emitted by |
|---|---|
| `AGENT_START` / `AGENT_FINISH` | every `Agent` run, including nested |
| `LOOP_STEP` | each iteration of an `LLMEngine` tool-calling loop |
| `MODEL_REQUEST` / `MODEL_RESPONSE` | every provider call |
| `TOOL_CALL` / `TOOL_RESULT` / `TOOL_ERROR` | every tool dispatch |
| `HIL_DECISION` | `HumanEngine` / `SupervisorEngine` decisions |

`Agent(verbose=True)` creates a private `Session(console=True)` for that
agent — useful for one-off debugging without wiring an explicit session.

## Synopsis

A `Session` does three things:

1. **Persists events** to an SQLite-backed `EventLog`. Every engine
   emits the same enum, so a single query returns a full per-run
   trace.
2. **Fans events out to exporters** in registration order — Console,
   `JsonFileExporter`, `OTelExporter`, custom sinks.
3. **Aggregates cost / tokens / latency** across the whole agent
   tree via `usage_summary()`, including transitive rollup from
   nested sub-agents.

Pass a `Session` once at the top-level agent. Nested agents (`Agent A`
with `Agent B` in `tools=[...]`) inherit it automatically; the graph
view shows the whole tree, the cost rollup includes every child.

## When to use it

- **You want any of**: cost tracking across multiple `agent(task)`
  calls, a JSON-line event log for offline analysis, OpenTelemetry
  spans, a graph view of an agent topology.
- **Production deployments** where the hot path can't block on disk
  / network — pair `batched=True` with `JsonFileExporter` /
  `OTelExporter`.
- **Multi-agent pipelines.** The session at the outermost agent
  collects events from every nested agent and tool call without any
  per-agent plumbing.
- **PII / sensitive-data scrubbing.** Pass a `redact=` callable that
  rewrites payloads before they reach exporters; the default
  `redact_on_error="strict"` fails closed if the redactor itself
  errors.  By default Session already runs
  `redact_secrets` — well-known credential shapes (`sk-...`,
  `ghp_...`, `AIza...`, JWT, `Bearer ...`) are stripped from every
  event payload before it leaves the bus.  Set
  `unsafe_log_payloads=True` (or `redact=None`) to disable it; pass
  your own `redact=` to replace it.

## When NOT to use it

- **Single one-off `agent(task)` call where you just want stdout.**
  Use `Agent(verbose=True)` instead — it's a private console
  session and saves you the import.
- **Real-time analytics with sub-millisecond latency requirements.**
  The default sync writer fits most workloads; for the hot path use
  `batched=True`. If even that's too slow, write a custom
  `EventExporter` that pushes to your own pipeline.

## Example

```python
from lazybridge import (
    Agent,
    JsonFileExporter,
    LLMEngine,
    Session,
)
from lazybridge.session import EventType


# 1) Dev-mode tracing — one flag.
sess = Session(console=True)
chat = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    session=sess,
    name="chat",
)
chat("hello")


# 2) Production shape — multi-sink, batched, redacted.
def mask_pii(payload: dict) -> dict:
    if "task" in payload:
        return {**payload, "task": payload["task"].replace("foo@bar.com", "[REDACTED]")}
    return payload


sess = Session(
    db="events.sqlite",
    batched=True,
    on_full="hybrid",                 # default; explicit for clarity
    exporters=[
        JsonFileExporter(path="events.jsonl"),
    ],
    redact=mask_pii,
)

researcher = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="research",
)
writer = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    name="write",
)
pipeline = Agent.chain(researcher, writer, session=sess)
pipeline("summarise AI trends")


# 3) Cost / token roll-up across the whole tree.
print(sess.usage_summary()["total"]["cost_usd"])


# 4) Drain the batched writer before reading the log.
sess.flush()
errors = sess.events.query(event_type=EventType.TOOL_ERROR)


# 5) Topology for a UI / report.
print(sess.graph.to_json())


# 6) OpenTelemetry — install lazybridge[otel].
from lazybridge.ext.otel import OTelExporter

sess.add_exporter(OTelExporter(endpoint="http://otelcol:4318"))
```

## Pitfalls

- **`Session(db=":memory:")` behaves like `Session()`** — both are
  in-memory. Pass a real filename to persist.
- **Exporter failures warn once per instance.** Subsequent failures
  from the same exporter are suppressed. While debugging a noisy
  exporter, wrap it in `CallbackExporter(fn=lambda e: print(e))` so
  you see every emission attempt.
- **`Agent(verbose=True)` creates a fresh private Session.** If you
  also pass `session=another`, `verbose` is ignored — the explicit
  session wins.
- **`batched=True` makes reads stale.** `session.events.query(...)`
  may not reflect events still in the writer queue; call
  `session.flush()` (or use `Session` as a context manager that
  auto-closes) before querying.
- **`on_full="drop"` was the pre-1.0.x default.** The new
  `"hybrid"` default holds critical events
  (`AGENT_*` / `TOOL_*` / `HIL_DECISION`) and only drops cheap
  telemetry (`LOOP_STEP` / `MODEL_REQUEST` / `MODEL_RESPONSE`)
  under saturation. Set `on_full="block"` if you need every event,
  no exceptions; set `on_full="drop"` to opt back into the old
  behaviour.
- **`redact_on_error="strict"` (default) drops the event** if the
  redactor raises or returns a non-dict. Use `"fallback"` only when
  you want the unredacted payload as a backup; the trade-off is the
  potential to log raw PII.
- **Default secret redaction is on.**  `Session()` with no `redact=`
  argument wires `redact_secrets` which masks `sk-...` /
  `ghp_...` / `AIza...` / JWT / `Bearer ...` shapes in payload
  strings.  It does *not* mask emails, phone numbers, or other PII —
  compose your own redactor on top if you need that.  Disable
  entirely with `unsafe_log_payloads=True` (or `redact=None`); pass
  your own `redact=` to replace it (LazyBridge does not stack the
  default in front of a user redactor).
- **Nested agents inherit `session=`** unless they pass their own.
  This is what gives you transitive cost rollup; pass an explicit
  `session=None` on a sub-agent only when you genuinely want it
  invisible.

## See also

- *Guides → Full → Exporters* (Phase 3) — the sinks that consume
  session events (`ConsoleExporter`, `JsonFileExporter`,
  `StructuredLogExporter`, `FilteredExporter`, `CallbackExporter`,
  `OTelExporter`).
- *Guides → Full → GraphSchema* (Phase 3) — the topology view
  exposed via `session.graph`.
- [Memory](memory.md) — separate concept (conversation context, not
  observability).
- [Agent](../basic/agent.md) — `session=` is a first-class kwarg;
  `verbose=True` is the convenience shortcut.
