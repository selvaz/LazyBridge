---
title: Why LazyBridge
---

# Why LazyBridge

Five things LazyBridge does differently — each grounded in code, not marketing.

---

## 1. Recursive composition

In LazyBridge, `Plan` is an engine. An `Agent(engine=Plan(...))` is a valid
`Tool`. That means pipelines nest with no special syntax at any depth — you
use the same `tools=[...]` parameter whether you're adding a plain function or
a ten-step sub-pipeline.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, Session, from_step

search    = Agent(engine=LLMEngine("gpt-5.4-mini"),      name="search")
summarise = Agent(engine=LLMEngine("gemini-2.5-pro"),    name="summarise")
writer    = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

research = Agent(
    engine=Plan(Step("search"), Step("summarise")),
    tools=[search, summarise], name="research",
)
article = Agent(
    engine=Plan(Step("research"),
                Step("write", context=from_step("research"))),
    tools=[research, writer], session=Session(),
)
print(article("AI agents in 2026").text())
```

Cost, token counts, and OpenTelemetry spans roll up automatically across all
levels via `Envelope.metadata.nested_*`. You pay no composition tax.

**Deep dive:** [Layered composition](concepts/layered-composition.md)

---

## 2. Plans fail at construction

Other frameworks surface wiring errors at runtime — after spending money on
LLM calls. LazyBridge raises `PlanCompileError` the moment you construct a
`Plan`, before any inference happens.

```python
from lazybridge import Agent, LLMEngine, Plan, Step, from_step

writer = Agent(engine=LLMEngine("claude-sonnet-4-6"), name="write")

# This raises PlanCompileError immediately — "search" is not in tools=[]
plan = Plan(
    Step("search"),
    Step("write", context=from_step("search")),
)
```

The compiler catches:

- Duplicate step names
- Tools referenced in `Step(target=...)` that are not in `tools=[]`
- Forward references to steps not yet declared
- Sentinels (`from_step`, `from_parallel`, `from_parallel_all`) pointing at
  incompatible neighbours in the same parallel band
- `from_parallel_all` on a step that is not a band-start
- Type drift between consecutive steps (structural mismatch)
- Malformed `routes=` / `routes_by=` declarations
- Invalid `after_branches=` references

All before a single token is generated.

**Deep dive:** [Plan guide](guides/full/plan.md)

---

## 3. One contract for everything

`Tool.wrap()` accepts anything that exposes a useful capability. There is one
interface whether the capability is a function, an agent, a Plan-backed
pipeline, an MCP server, or a native provider tool:

| Capability | How it enters | What the agent sees |
|---|---|---|
| Python function | `Tool.wrap(fn)` or `tools=[fn]` | `Tool` |
| `Agent` | `agent.as_tool()` or `tools=[agent]` | `Tool` |
| `Agent(engine=Plan)` | `tools=[pipeline]` | `Tool` |
| MCP server | `tools=[MCP.stdio(...)]` | `Tool` |
| Native provider tool | `tools=[NativeTool(...)]` | `Tool` |
| Guarded variant | `agent.as_tool(verify=judge)` | `Tool` |

The LLM engine sees the same schema regardless of what backs the tool. Swap
a stub function for a full sub-pipeline without touching the outer agent.

**Deep dive:** [Everything is a tool](concepts/everything-is-a-tool.md)

---

## 4. Quality control at every node

Structured output, runtime validation, and cross-model verification compose
at any level of the pipeline — not just at the top.

```python
from pydantic import BaseModel
from lazybridge import Agent, LLMEngine

class Report(BaseModel):
    title: str
    summary: str
    word_count: int

judge = Agent(engine=LLMEngine("gemini-2.5-pro"), name="judge")

writer = Agent(
    engine=LLMEngine("claude-sonnet-4-6"),
    output=Report,                          # schema-validated; re-prompts on failure
    output_validator=lambda r: r.word_count > 50,  # runtime check
    verify=judge,                           # cross-model quality gate
    max_verify=3,                           # up to 3 retry cycles
    name="write",
)
```

`output=` validates the schema and re-prompts automatically on parse failure.
`output_validator=` runs an arbitrary Python check after parsing. `verify=`
passes the output to a second agent for quality assessment; feedback flows
via context, never onto the original task, and loops up to `max_verify` times.
All three stack.

**Deep dive:** [verify= guide](guides/mid/verify.md)

---

## 5. Observability is native

Session and event logging are on by default. OpenTelemetry export with
GenAI semantic conventions (`gen_ai.*` attributes) requires one line:

```python
from lazybridge import Agent, LLMEngine, Session
from lazybridge.ext.otel import OTelExporter

agent = Agent(
    engine=LLMEngine("claude-sonnet-4-6"),
    session=Session(exporters=[OTelExporter()]),
)
```

Every run emits spans carrying `gen_ai.system`, `gen_ai.request.model`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `nesting_level`,
and parent-child span links. Cost rolls up across nested agents.
SQLite-backed `Store` gives you a local event log with back-pressure
policies and batching. Both are opt-out, not opt-in.

**Deep dive:** [Session guide](guides/mid/session.md)

---

## Next steps

[**Quickstart →**](quickstart.md){ .md-button .md-button--primary }
[Comparison with other frameworks](comparison.md){ .md-button }
[Layered composition deep-dive](concepts/layered-composition.md){ .md-button }
