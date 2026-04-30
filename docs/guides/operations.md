# Running LazyBridge in production

A focused checklist for operators. Each item is a knob the framework
already exposes; this page tells you which to flip and why. For the
fine-grained per-topic guidance, follow the cross-links.

## 1 · Observability — OTel GenAI conventions

LazyBridge's `OTelExporter` emits spans conforming to the
[OpenTelemetry Semantic Conventions for Generative AI][otel-genai].
Existing GenAI dashboards (Datadog, Honeycomb, Grafana Tempo) render
LazyBridge traces unchanged.

[otel-genai]: https://opentelemetry.io/docs/specs/semconv/gen-ai/

```python
from lazybridge import Agent, Session, JsonFileExporter
from lazybridge.ext.otel import OTelExporter

sess = Session(
    db="events.sqlite",
    batched=True,                              # non-blocking emit
    exporters=[
        JsonFileExporter("events.jsonl"),
        OTelExporter(endpoint="http://otelcol:4318"),
    ],
)
```

Span hierarchy:

```
invoke_agent <agent_name>            (root per Agent.run)
  ├─ chat <model>                    (one per LLM round-trip)
  └─ execute_tool <tool_name>        (correlated by tool_use_id)
        └─ invoke_agent <inner>      (when the tool is itself an Agent)
```

Cross-agent parenting works automatically through OTel contextvars —
no run-id chaining required. See the [exporters guide](exporters.md).

## 2 · Back-pressure — `on_full="hybrid"`

When `Session(batched=True)`, the writer queue saturates if exporters
fall behind producers. Three policies:

| Policy | Behaviour | When to pick |
|---|---|---|
| **`"hybrid"`** *(default)* | Block on `AGENT_*` / `TOOL_*` / `HIL_DECISION`; drop `LOOP_STEP` / `MODEL_REQUEST` / `MODEL_RESPONSE` | Production default. Audit-relevant events never silently disappear; cheap telemetry doesn't add producer latency. |
| `"block"` | Block unconditionally | Compliance workloads — every event must persist. |
| `"drop"` | Never block; drop on saturation with a doubling-interval warning | Telemetry must not block production traffic; lossy traces acceptable. |

```python
from lazybridge.session import EventType

# Override the critical-event set if your shape is different.
sess = Session(
    batched=True,
    on_full="hybrid",
    critical_events={EventType.AGENT_FINISH, EventType.TOOL_ERROR},
)
```

`session.flush(timeout=5.0)` drains the writer before reads or shutdown.
`session.close()` (or using the `Session` as a context manager) flushes
and releases SQLite connections deterministically.

## 3 · Reliability knobs on `Agent`

The framework's defaults are dev-friendly. Production deployments
typically opt into these:

```python
from lazybridge import Agent

agent = Agent(
    "claude-opus-4-7",
    tools=[search],
    timeout=30.0,                # total deadline for run()
    max_retries=3,               # provider transient-error retries
    retry_delay=1.0,             # base for exponential backoff (±10% jitter)
    cache=True,                  # prompt caching where supported
    fallback=Agent("gpt-5"),     # provider redundancy on hard failure
    output_validator=validate,   # custom post-parse check
    max_output_retries=2,        # structured-output retry-with-feedback
)
```

The fallback receives the primary's failure mode in `Envelope.context`
(e.g. `"Previous attempt failed with RateLimitError: ..."`) so it can
adapt its strategy. See [the Agent guide](agent.md).

## 4 · Crash resume — `Plan` checkpoints

`Plan` checkpoints to `Store` via `compare_and_swap` after every step.
Two policies for concurrent runs that share a `checkpoint_key`:

| Policy | Use for |
|---|---|
| `on_concurrent="fail"` *(default)* | Single-writer semantics; second concurrent run raises `ConcurrentPlanRunError`. Pair with `resume=True` for graceful crash-resume. |
| `on_concurrent="fork"` | Fan-out workflows — each run claims a per-run-uid suffixed key. No single shared checkpoint, so `resume=True` is not allowed in this mode. |

Atomicity inside a parallel band: if any branch errors, no `writes`
are applied — a future resume re-runs the whole band cleanly instead
of double-applying succeeded siblings' side-effects. See
[Checkpoint & resume](checkpoint.md).

## 5 · Memory hygiene

`Memory.summarizer_timeout` (default **30 s**) keeps a hung LLM
summariser from blocking the agent's tool loop. On deadline the
keyword-extraction fallback runs and a one-shot warning surfaces.

Compression itself runs OUTSIDE the internal lock — concurrent
`memory.add()` calls keep progressing while a slow summariser is in
flight.

```python
from lazybridge import Agent, Memory

mem = Memory(
    strategy="summary",
    max_tokens=3000,
    summarizer=cheap_agent,
    summarizer_timeout=15.0,
)
```

See [the Memory guide](memory.md).

## 6 · MCP cache TTL

`MCPServer` caches the discovered tool list for `cache_tools_ttl`
seconds (default **60 s**). After expiry the next call re-fetches
from the upstream server. Override paths:

```python
fs = MCP.stdio("fs", command="...", cache_tools_ttl=300.0)  # 5 min
fs = MCP.stdio("fs", command="...", cache_tools_ttl=None)   # always-cache
fs.invalidate_tools_cache()                                 # manual flush
```

See [the MCP recipe](../recipes/mcp.md).

## 7 · Tool-call parse errors

When a model emits a malformed tool-call JSON blob, the engine emits a
structured `TOOL_ERROR` (`type="ToolArgumentParseError"`) **before** the
tool runs:

```python
{
    "tool":            "search",
    "tool_use_id":     "call-1",
    "type":            "ToolArgumentParseError",
    "error":           "Tool 'search' received malformed JSON arguments ...",
    "parse_error":     "Expecting property name enclosed in double quotes...",
    "raw_arguments":   "{not valid",
}
```

The model sees the **real** failure on the next turn (its JSON output
was malformed) and self-corrects, rather than failing later with a
misleading "missing required field" message.

## 8 · Cost roll-up

`Envelope.metadata.nested_*` aggregates token / cost telemetry across
nested Agent-of-Agents trees. The outer envelope reports total
pipeline spend without double-counting:

```python
env = pipeline("draft a one-pager")
total_cost = env.metadata.cost_usd + env.metadata.nested_cost_usd
total_in   = env.metadata.input_tokens + env.metadata.nested_input_tokens
total_out  = env.metadata.output_tokens + env.metadata.nested_output_tokens
```

`Session.usage_summary()` does the same aggregation for an entire
session — by agent, by run, and totals.

## 9 · CI hardening

The repo ships with:

* **`release.yml`** — PyPI Trusted Publishing on `v*.*.*` tags, with
  tag-vs-`pyproject` version verification.
* **`codeql.yml`** — weekly scheduled SAST + per-PR.
* **`dependabot.yml`** — weekly action + pip updates; provider SDK
  major versions intentionally pinned (`anthropic`, `openai`,
  `google-genai`, `litellm`, `pydantic`).
* **`test.yml`** — pre-commit job + lint + typecheck + unit tests on
  Python 3.11 / 3.12 / 3.13.
* **Coverage gate** — `fail_under = 70` covering core + cross-cutting
  `ext.{otel,mcp,hil,planners,evals}`. Domain-specific extensions
  (`stat_runtime`, `data_downloader`, …) are tested separately under
  `tests/unit/ext/`.

## See also

* [Session & tracing](session.md)
* [Exporters](exporters.md)
* [Memory](memory.md)
* [Plan](plan.md) · [Checkpoint & resume](checkpoint.md)
* [MCP recipe](../recipes/mcp.md)
