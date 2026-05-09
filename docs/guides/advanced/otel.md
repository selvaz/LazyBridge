# OpenTelemetry

`OTelExporter` emits LazyBridge `Session` events as OpenTelemetry
spans conforming to the GenAI Semantic Conventions
(`gen_ai.system`, `gen_ai.usage.input_tokens`,
`gen_ai.tool.call.id`, …). Dashboards built for the standard
(Datadog GenAI, Honeycomb GenAI, Grafana Tempo) render LazyBridge
traces without translation.

This page is the deep dive: span hierarchy, attribute names, tracer
provider lifecycle, and the in-memory exporter pattern for tests.
For exporter basics see [Exporters](../full/exporters.md).

## Signature

```python
# Install — opt-in extra.
pip install "lazybridge[otel]"


from lazybridge import Session
from lazybridge.ext.otel import OTelExporter


OTelExporter(
    *,
    endpoint=None,                 # OTLP HTTP endpoint string
    exporter=None,                 # custom OTel exporter instance (overrides endpoint)
    batch=True,                    # True → BatchSpanProcessor; False → SimpleSpanProcessor
)


sess = Session(
    db="events.sqlite",
    batched=True,                  # Session-level back-pressure for the hot path
    exporters=[OTelExporter(endpoint="http://otelcol:4318")],
)
```

## Synopsis

The exporter takes raw `Session` events and translates each into the
appropriate OTel span. It manages span lifecycles (open on
`AGENT_START` / `MODEL_REQUEST` / `TOOL_CALL`, close on the matching
finish event), attaches them to the OTel context so nested operations
inherit the right parent, and detaches when they close.

### Span hierarchy

```text
invoke_agent  <agent_name>          (root for one Agent.run)
  ├─ chat       <model>             (one per LLM round-trip)
  └─ execute_tool <tool_name>       (one per tool invocation)
        └─ invoke_agent <inner>     (when the tool is itself an Agent)
```

Tool spans run as children of the agent span and close on
`TOOL_RESULT` / `TOOL_ERROR` (correlated by `tool_use_id`).
Cross-agent parenting works automatically: the inner agent's events
are emitted on the same asyncio context as the outer tool span, so
OTel's contextvars-based propagation makes the inner `invoke_agent`
span a child of the outer `execute_tool` span without any explicit
run-id chaining.

### GenAI Semantic Convention attributes

| Attribute | Source field | Where it appears |
|---|---|---|
| `gen_ai.system` | provider name (`"anthropic"`, `"openai"`, …) | `chat` spans |
| `gen_ai.operation.name` | `"chat"` / `"execute_tool"` / `"invoke_agent"` | every span |
| `gen_ai.request.model` | configured model | `chat` spans |
| `gen_ai.response.model` | actual model the provider replied with | `chat` spans |
| `gen_ai.usage.input_tokens` | input tokens for this round-trip | `chat` spans |
| `gen_ai.usage.output_tokens` | output tokens for this round-trip | `chat` spans |
| `gen_ai.response.finish_reasons` | normalised stop reason | `chat` spans |
| `gen_ai.tool.name` | tool name | `execute_tool` spans |
| `gen_ai.tool.call.id` | `tool_use_id` for correlation | `execute_tool` spans |
| `gen_ai.agent.name` | wrapping agent's name | `invoke_agent` spans |

### LazyBridge-specific attributes

These have no GenAI equivalent (yet). They're prefixed `lazybridge.*`
so an operator can filter on them deterministically without
mistaking them for a future GenAI rename.

| Attribute | Meaning |
|---|---|
| `lazybridge.run_id` | UUID identifying the agent run (one per `Agent.run`) |
| `lazybridge.cost_usd` | Cost in USD reported by the provider for this round-trip |
| `lazybridge.turn` | Loop iteration index inside `LLMEngine`'s tool-calling loop |
| `lazybridge.branch_id` | Parallel-branch step name when emitted from a band |

## When to use it

- **Distributed tracing** — you already run an OTel collector
  (Datadog, Honeycomb, Tempo, Jaeger) and want LazyBridge traces
  to land in the same dashboards.
- **Multi-service correlation** — cross-agent calls (Agent A as a
  tool inside Agent B) span pretty across worker thread / asyncio
  boundaries, since OTel contextvars propagate the active span
  automatically.
- **Cost / token attribution** — the `gen_ai.usage.*` and
  `lazybridge.cost_usd` attributes let you slice spend by model /
  agent / tool in the same dashboard you use for latency and
  errors.
- **Compliance / audit trails** — every tool call shows up as its
  own span with correlation id and structured arguments.

## When NOT to use it

- **You don't have an OTel pipeline.** Use
  [`JsonFileExporter`](../full/exporters.md) and load the
  resulting JSONL into pandas / your preferred tool. Standing up a
  collector just for one app's traces is overkill.
- **You want to query history programmatically.**
  `session.events.query(...)` reads the SQLite-backed `EventLog`
  directly — no collector required. OTel is for *push* streams to
  external systems.
- **Single-process scripts.** A console exporter (`Session(console=True)`
  or `Agent(verbose=True)`) is enough; OTel adds infrastructure
  weight you don't need.

## Example

```python
from lazybridge import Agent, LLMEngine, Session
from lazybridge.ext.otel import OTelExporter


# 1) Production: OTLP HTTP endpoint, batched span processor (default).
sess = Session(
    db="events.sqlite",
    batched=True,                      # session-level back-pressure
    exporters=[OTelExporter(endpoint="http://otelcol:4318")],
)
agent = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    session=sess,
)
agent("hello")
sess.flush()                           # drain the batched writer before exit
sess.close()                           # also flushes the OTel batch processor


# 2) Tests: in-memory exporter so each span is captured synchronously.
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


def test_agent_emits_spans():
    memory_exporter = InMemorySpanExporter()
    sess = Session(
        exporters=[
            OTelExporter(exporter=memory_exporter, batch=False),
        ],
    )

    agent = Agent(
        engine=LLMEngine("claude-opus-4-7"),
        session=sess,
    )
    agent("test prompt")
    sess.close()

    spans = memory_exporter.get_finished_spans()
    assert any(s.name.startswith("invoke_agent") for s in spans)


# 3) Multiple exporters in one session — OTel + JSON file + console alerts.
from lazybridge import (
    CallbackExporter,
    ConsoleExporter,
    EventType,
    FilteredExporter,
    JsonFileExporter,
)


def alert_on_error(event: dict) -> None:
    print(f"ALERT: {event}")


sess = Session(
    db="events.sqlite",
    batched=True,
    exporters=[
        JsonFileExporter(path="run.jsonl"),
        OTelExporter(endpoint="http://otelcol:4318"),
        FilteredExporter(
            inner=CallbackExporter(fn=alert_on_error),
            event_types={EventType.TOOL_ERROR, EventType.AGENT_FINISH},
        ),
    ],
)
```

## Pitfalls

- **`batch=True` (default) buffers spans.** Spans from a fast-
  finishing run may not be flushed by the time the process exits.
  Always `sess.close()` (or use `Session` as a context manager)
  before exit so the BatchSpanProcessor drains.
- **`batch=False` ↔ `SimpleSpanProcessor`** is the right choice
  for tests against an `InMemorySpanExporter` — every span is
  flushed synchronously on close. Don't use it in production:
  every span emit blocks the engine on the network round-trip.
- **Per-instance `TracerProvider`.** Each `OTelExporter` creates
  its own `TracerProvider`. The provider is also installed
  globally as a best-effort default (so unrelated OTel-aware
  code in the same process picks it up), but two
  `OTelExporter` instances don't share state — each manages its
  own in-flight span registry. For tests with multiple exporters,
  pass distinct `exporter=` arguments.
- **Stale spans on cancellation.** If a run is cancelled before
  `AGENT_FINISH` fires, the corresponding span stays open in the
  registry. Call `sess.close()` (which calls `OTelExporter.close()`)
  to force-flush any spans still open — useful when the cancel
  is graceful but the finally-block can't reach the finish emit.
- **Custom resource attributes** (service.name, deployment.env)
  aren't set by the exporter. Configure them on your own
  `TracerProvider` and pass it through `exporter=` if you want
  them on every span — `endpoint=` on the exporter constructor
  builds a default provider with no resource attributes.
- **Native-tool calls don't appear as `execute_tool` spans.**
  Those happen server-side at the provider; LazyBridge sees them
  as part of the model's response, so they roll into the parent
  `chat` span rather than getting their own. If you want fine-
  grained native-tool tracing, query the provider's own dashboard.
- **`session.events.query(...)` reads stale data when batched.**
  The SQLite write is batched separately from the OTel emit;
  call `sess.flush()` before reading the local event log.

## See also

- [Exporters](../full/exporters.md) — the broader exporter surface
  including the four core sinks (Console, JsonFile, StructuredLog,
  Callback) and `FilteredExporter`.
- [Session](../mid/session.md) — the bus that fans events into
  exporters; covers `batched=`, `on_full=`, `redact=`, and the
  `EventLog` query surface.
- [GraphSchema](../full/graph-schema.md) — agent topology view;
  complements OTel (graph = static structure, OTel spans = live
  trace).
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
  — the upstream spec the exporter conforms to.
