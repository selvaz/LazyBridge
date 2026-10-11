---
title: Why LazyBridge
---

# Why LazyBridge

Claude Code and Codex run as engines inside the same Python Agent contract.
One can write while the other reviews, with shared tools and application state.
That contract also includes provider APIs, deterministic Plans and human decisions.

The design ideas below explain how those pieces compose. They are API behavior,
not a claim of measured savings or superiority over another framework.

## 1. Recursive composition

`Plan` is an engine. A named `Agent(engine=Plan(...))` can be a Tool of another
Agent. Functions, model-driven workers and reusable pipelines therefore enter
the same `tools=[...]` surface.

```python
from lazybridge import Agent, ClaudeCodeEngine, LLMEngine, Plan, Step, Session, from_step

search = Agent(engine=LLMEngine("gpt-5.4-mini"), name="search")
summarise = Agent(engine=LLMEngine("gemini-2.5-pro"), name="summarise")
writer = Agent(engine=ClaudeCodeEngine(), name="write")

research = Agent(
    engine=Plan(Step("search"), Step("summarise")),
    tools=[search, summarise], name="research",
)
article = Agent(
    engine=Plan(Step("research"), Step("write", context=from_step("research"))),
    tools=[research, writer], name="article", session=Session(),
)
result = article("Explain deployment policy")
assert result.ok
print(result.text())
```

Nested token and cost metadata roll up through `Envelope.metadata.nested_*`.
The outer run measures its own elapsed time; it does not sum parallel latency.
Child errors remain visible through tool results and Session events. Supplying
a Session lets nested agents inherit the event/trace context when they have no
explicit Session of their own.

**Deep dive:** [Layered composition](concepts/layered-composition.md).

## 2. Plans validate before execution

Plan construction checks structural wiring. Binding it to an Agent supplies the
tool map for tool-name and compatible typed-handoff checks. These errors can
be found before invoking a model; they are not a proof of every runtime value
or external side effect.

```python
from lazybridge import Agent, Plan, PlanCompileError, Step
from lazybridge.testing import MockAgent

writer = MockAgent("draft", name="write")
plan = Plan(Step("search"), Step("write"))
try:
    Agent(engine=plan, tools=[writer], name="article")
except PlanCompileError as exc:
    print("Missing search tool:", exc)
else:
    raise AssertionError("Expected the missing-tool check")
```

The compiler checks declared constraints, including:

- Duplicate step names and missing tool targets once tools are bound.
- Invalid references to unavailable steps.
- Incompatible sentinels within a parallel band.
- Invalid placement of `from_parallel_all`.
- Structurally incompatible typed handoffs when types can be derived.
- Malformed route declarations and invalid reconvergence references.

Runtime parsing and validation still matter for model output, dynamic tool data
and branches that cannot be checked statically. Other frameworks also offer
structural checks; this describes LazyBridge's particular boundary.

**Deep dive:** [Plan guide](guides/full/plan.md).

## 3. One application Tool contract

`Tool.wrap()` normalizes application capabilities. Replace a stub function with
a named sub-agent or a Plan-backed pipeline while keeping the outer tool contract.

| Capability | How it enters | Execution boundary |
|---|---|---|
| Python function | Tool.wrap(fn) or tools=[fn] | Python Tool |
| Named Agent | agent.as_tool() or tools=[agent] | Nested Agent through Tool |
| Plan-backed Agent | tools=[pipeline] | Deterministic engine through Tool |
| MCP catalogue | LazyTools provider in tools | Connector-expanded Python Tools |
| Verified tool view | agent.as_tool(verify=judge) | Tool with bounded review |
| Provider-native capability | LLMEngine.native_tools | Provider runtime, separate from Tool.wrap |
| Coding runtime built-in | Engine policy/config | Claude built-in or Codex sandbox action |

Claude Code exposes Python Tools through its temporary in-process MCP adapter;
Codex exposes them as App Server dynamic tools. Both dispatch through Tool.run.
Native filesystem and shell permissions do not sandbox a Python tool's side effects.

MCP, connectors, code support and repository jobs belong to
[LazyTools](https://tools.lazybridge.com/). See its
[code bridge](https://tools.lazybridge.com/code-bridge/) for detached work,
durable tickets and opposite-engine review; Core does not duplicate those APIs.

**Deep dive:** [Everything is a tool](concepts/everything-is-a-tool.md).

## 4. Quality control at every node

Structured output, runtime validation and reviewer feedback can compose at any
level, including a critical sub-agent exposed as a Tool.

```python
from pydantic import BaseModel
from lazybridge import Agent, ClaudeCodeEngine, CodexEngine

class Report(BaseModel):
    title: str
    summary: str
    word_count: int

def validate_report(report: Report) -> Report:
    if report.word_count <= 50:
        raise ValueError("Expand the report beyond 50 words")
    return report

judge = Agent(name="judge", engine=CodexEngine(
    system='Start with "approved" or "rejected: <reason>". Check completeness.',
))
writer = Agent(
    name="write", engine=ClaudeCodeEngine(), output=Report,
    output_validator=validate_report, verify=judge, max_verify=3,
)
result = writer("Write a report on weekday deployment; include an accurate word count.")
assert isinstance(result.payload, Report)
```

`output=` parses and validates the declared schema, with repair attempts bounded
by `max_output_retries`. `output_validator=` must return the validated value or
raise on failure; returning a boolean would replace the payload, not reject it.

`verify=` passes output to a judge. Rejection feedback goes through context,
preserving the original task. `max_verify=3` means at most three producer attempts,
each reviewed. Exhaustion returns the last output without an approval flag:
`result.ok` means execution succeeded. Use deterministic tests and explicit
acceptance records before publishing or merging a result.

Choose Codex to review a Claude producer, or Claude to review a Codex producer.
Core accepts other judges too; opposite-engine selection is explicit policy.

**Deep dive:** [verify=](guides/mid/verify.md) and
[Claude Code writes, Codex reviews](guides/full/claude-writes-codex-reviews.md).

## 5. Observability is part of the contract

Every call returns an Envelope with payload, error and usage metadata. Session
logging is opt-in: supply a Session, or use verbose=True for a console Session.
Store holds application records; it is not the Session event log.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from lazybridge import Agent, ClaudeCodeEngine, JsonFileExporter, Session

with TemporaryDirectory() as directory:
    log = Path(directory) / "events.jsonl"
    session = Session(exporters=[JsonFileExporter(path=str(log))])
    agent = Agent(name="observed", engine=ClaudeCodeEngine(), session=session)
    result = agent("Report readiness for review")
    session.close()
    assert result.ok and log.read_text(encoding="utf-8")
```

Session exporters can write JSON Lines, structured logs or application callbacks.
Session(db=...) persists events in SQLite; batching and queue-full policy belong
to Session. Store(db=...) separately persists key/value application records.

The optional OpenTelemetry exporter emits GenAI spans for agents, model calls
and tool dispatch, with nested parenting and reported token/cost attributes.
Use the otel extra and choose an exporter/backend explicitly. This example uses
an in-memory sink, with no network export:

```python
from lazybridge import Agent, Session
from lazybridge.testing import MockAgent

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:
    print("Install lazybridge[otel] for OpenTelemetry export")
else:
    from lazybridge.ext.otel import OTelExporter
    sink = InMemorySpanExporter()
    session = Session(exporters=[OTelExporter(exporter=sink, batch=False)])
    worker = MockAgent("observed", name="worker")
    pipeline = Agent.chain(worker, name="trace-demo", session=session)
    assert pipeline("Run the trace example").ok
    session.close()
```

**Deep dive:** [Session](guides/mid/session.md), [exporters](guides/full/exporters.md)
and [OpenTelemetry](guides/advanced/otel.md).

## Continuity and engine swaps

Keep the same tools, Memory, Store and Session references when constructing an
Agent with a different engine. Native continuity is a separate choice:
`persist_session`/`session_id` for Claude, `persist_thread`/`thread_id` for Codex,
or a `session_alias` bound by SessionRegistry to engine kind and cwd.

Resumed native history omits repeated Memory priming while retaining explicit
context. A runtime session slot is not a durable alias, and a Claude transcript
cannot be transferred into a Codex thread simply by swapping engines.

See [coding agents](guides/full/coding-agents.md) and
[SessionRegistry](guides/full/session-registry.md). The LazyTools quota routing
benchmark is **in progress**.

## Next steps

[**Quickstart →**](quickstart.md){ .md-button .md-button--primary }
[Comparison with other frameworks](comparison.md){ .md-button }
[Layered composition](concepts/layered-composition.md){ .md-button }
