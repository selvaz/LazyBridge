# Exporters

The sinks that consume `Session` events. Five built-ins ship from
the core package; an OpenTelemetry exporter ships from
`lazybridge.ext.otel`. Compose them — most production setups use
two or three at once.

## Signature

```python
# Protocol every exporter satisfies.
class EventExporter:
    def export(self, event: dict) -> None: ...
    def close(self) -> None: ...           # optional; called by Session.close()


# Built-ins from core (lazybridge package).
from lazybridge import (
    CallbackExporter,
    ConsoleExporter,
    FilteredExporter,
    JsonFileExporter,
    StructuredLogExporter,
    EventType,
)

# Every core exporter has a keyword-only constructor.
CallbackExporter(*, fn)                                  # fn: Callable[[dict], None]
ConsoleExporter(*, stream=None)                          # defaults to sys.stdout when None
FilteredExporter(*, inner, event_types)                  # combinator: forward only matching events
JsonFileExporter(*, path)                                # JSON-lines append (one event per line)
StructuredLogExporter(*, logger_name="lazybridge")


# Built-in from ext (lazybridge[otel] extras).
from lazybridge.ext.otel import OTelExporter
OTelExporter(*, endpoint=None, exporter=None, batch=True)
```

Wire any list of exporters into a `Session(exporters=[...])`.

## Synopsis

A `Session` fans every event into all registered exporters in
registration order. Each event is a `dict` with at minimum
`event_type`, `session_id`, `run_id` (possibly `None`); engine-
specific fields are merged in by the emitter.

The full `EventType` enum (`lazybridge.session.EventType`):

| Member | Emitted by |
|---|---|
| `AGENT_START` / `AGENT_FINISH` | every `Agent` run, including nested |
| `LOOP_STEP` | each iteration of an `LLMEngine` tool-calling loop |
| `MODEL_REQUEST` / `MODEL_RESPONSE` | every provider call |
| `TOOL_CALL` / `TOOL_RESULT` / `TOOL_ERROR` / `TOOL_TIMEOUT` | every tool dispatch (with `TOOL_TIMEOUT` carrying `timeout_s`) |
| `HIL_DECISION` | one per `HumanEngine` / `SupervisorEngine` decision |

The five core exporters cover the dev / prod / custom surface:

| Exporter | What it does | When |
|---|---|---|
| `ConsoleExporter` | Pretty-prints events to stdout | Dev — same as `Agent(verbose=True)` |
| `JsonFileExporter` | Appends events as JSON Lines | Durable run logs for offline analysis |
| `StructuredLogExporter` | Emits via Python `logging` | Integrates with existing log pipelines |
| `CallbackExporter` | Calls a user-supplied function | Custom sinks: Slack alerts, Prometheus, your own DB |
| `FilteredExporter` | Wraps another exporter, forwarding only matching event types | Layer onto the others to scope the volume |

The OpenTelemetry exporter is a separate install — it emits spans
conforming to the OpenTelemetry GenAI Semantic Conventions
(`gen_ai.system`, `gen_ai.usage.input_tokens`,
`gen_ai.tool.call.id`, …) so dashboards built for the standard
render LazyBridge traces without translation. Span hierarchy mirrors
the run:

- `invoke_agent <name>` (root per `agent.run`)
    - `chat <model>` (one per LLM round-trip)
    - `execute_tool <tool>` (one per tool invocation, correlated by `tool_use_id`)

Cross-agent parenting works automatically through OTel contextvars
— an inner agent invoked through a tool becomes a descendant of the
outer tool span, no run-id chaining required.

For high-throughput emit paths, pair `Session(batched=True,
on_full="hybrid")` with the slow exporters (OTel, JSON-file).

`OTelExporter` defaults to `batch=True`, which wraps the underlying
OTLP exporter in `BatchSpanProcessor`. Set `batch=False` to use
`SimpleSpanProcessor` instead — primarily useful in tests against an
in-memory exporter, where you want each span flushed synchronously
on close.

## When to use which

- **`ConsoleExporter`** — local development, REPL inspection. The
  `Agent(verbose=True)` shortcut creates a private session with
  this exporter wired in; you don't need to construct it manually
  unless you also want other sinks.
- **`JsonFileExporter`** — durable per-run logs you can grep,
  pipe into jq, or load into pandas. The cheap default for any
  long-running production agent.
- **`StructuredLogExporter`** — integrates with existing
  `logging` pipelines. Lets central log management
  (CloudWatch / GCP Logging / ELK / Loki) ingest agent events
  through the same path as your application logs.
- **`CallbackExporter`** — anything else: alerting, custom DB
  writes, real-time UIs, metrics. Keep the callback fast; slow
  callbacks block the session unless `batched=True`.
- **`FilteredExporter`** — wrap one of the above when you only
  want a slice of events at that sink. Common pattern: an alert
  callback only sees `TOOL_ERROR` and `AGENT_FINISH`.
- **`OTelExporter`** — distributed tracing. The right answer when
  you already have an OTel collector and want LazyBridge traces
  in the same dashboards as the rest of your services.

## When NOT to use exporters

- **You don't have a `Session`.** Exporters are sinks for session
  events; an agent without a `session=` (and without
  `verbose=True`) emits nothing.
- **You want raw queryable history rather than push streams.**
  `Session.events.query(...)` reads the SQLite-backed `EventLog`
  directly — no exporter required. Use exporters when something
  *external* needs the events.
- **Your hot path can't tolerate the I/O cost.** Default
  `Session(batched=False)` blocks the engine on every export.
  Either batch, or pick exporters whose `export(event)` is
  microsecond-cheap.

## Example

```python
from lazybridge import (
    Agent,
    CallbackExporter,
    ConsoleExporter,
    FilteredExporter,
    JsonFileExporter,
    LLMEngine,
    Session,
)
from lazybridge.session import EventType


def alert_pagerduty(event: dict) -> None:
    """Fire a PagerDuty incident for tool errors and finish events."""
    ...


# 1) Dev — single console exporter, blocking emit.
sess = Session(console=True)
agent = Agent(
    engine=LLMEngine("gpt-5.4-mini"),
    session=sess,
)
agent("hello")


# 2) Production — batched writer + multiple exporters + filtered alerts.
def alert_pagerduty(event: dict) -> None:
    ...


sess = Session(
    db="events.sqlite",
    batched=True,
    on_full="hybrid",
    exporters=[
        JsonFileExporter(path="run.jsonl"),
        FilteredExporter(
            CallbackExporter(fn=alert_pagerduty),
            event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
        ),
    ],
)

researcher = Agent(engine=LLMEngine("gpt-5.4-mini"), name="research")
writer = Agent(engine=LLMEngine("gpt-5.4-mini"), name="write")
pipeline = Agent.chain(researcher, writer, session=sess)
pipeline("AI trends")

# Drain the writer before exit so JsonFileExporter / OTel get every
# in-flight event.
sess.flush()


# 3) OpenTelemetry — install lazybridge[otel].
from lazybridge.ext.otel import OTelExporter

sess.add_exporter(OTelExporter(endpoint="http://otelcol:4318"))
pipeline("more AI trends")
sess.close()


# 4) Custom — a CallbackExporter feeding a Slack channel.
def slack_post(event: dict) -> None:
    if event["event_type"] == EventType.AGENT_FINISH:
        post_to_slack(f"agent {event.get('agent_name')} finished")


sess = Session(
    exporters=[
        FilteredExporter(
            CallbackExporter(fn=slack_post),
            event_types={EventType.AGENT_FINISH},
        ),
    ],
)
```

## Pitfalls

- **Slow exporters block the engine when `Session(batched=False)`
  (the default).** Set `batched=True` for any exporter doing
  network I/O — JSON-file disk writes are usually fine
  unbatched; OTel network calls and external HTTP callbacks are
  not.
- **Exporter exceptions warn once per instance.** Subsequent
  failures from the same exporter are silently suppressed. While
  debugging a noisy exporter, wrap it in
  `CallbackExporter(fn=lambda e: print(e))` so you see every
  emission attempt.
- **`OTelExporter` keeps a per-instance tracer rooted in its own
  `TracerProvider`** so multiple exporters in one process don't
  fight. The provider is also installed globally as a best-effort
  default; you can supply your own and pass an in-memory exporter
  for tests.
- **Stale reads on batched sessions.** When
  `Session(batched=True)`, `session.events.query(...)` may
  return rows that don't yet include the most recent events —
  the writer hasn't drained. Call `session.flush()` (or use
  `Session.close()`) before querying.
- **Exporters are a per-session list.** A nested agent that
  inherits the parent's session inherits the parent's exporters.
  If a sub-agent should be invisible, give it its own
  `session=Session()`.
- **`FilteredExporter` only filters; it doesn't transform.** Use
  `CallbackExporter(fn=...)` if you need to rewrite events
  before forwarding (or wrap multiple sinks in a single callback
  that does the rewrite + dispatch).

## See also

- [Session](../mid/session.md) — the bus that fans events into
  exporters; covers `batched=`, `on_full=`, `redact=`,
  `usage_summary()`.
- [GraphSchema](graph-schema.md) — the topology view, separate
  from the event stream; populated as agents register.
- *Guides → Advanced → OTel* (Phase 3c) — the deeper
  OpenTelemetry surface (tracer providers, custom resource
  attributes, in-memory exporters for tests).
