# Progressive complexity

> Simple use cases stay simple. Complex workflows are possible without
> changing the core mental model.

LazyBridge is designed so the framework grows in complexity only when
**your problem grows**. You learn the next rung when you need it, not
before. Every rung is additive: the code you wrote at level 1 still works
at level 9.

## The ladder

```text
   1. Single agent
   2. Agent with tools
   3. Structured output
   4. Sequential chain               ──┐
   5. Parallel fan-out                 ├── multi-agent composition
   6. Agent as tool of another agent ──┘
   7. Deterministic Plan
   8. Plan with parallel band + routing
   9. Checkpoint + resume
  10. Human-in-the-loop gate
  11. Session, exporters, OpenTelemetry
  12. Custom engine / custom provider
```

You can stop at any rung. Most production systems live between rungs 4 and
9. Rung 12 exists; you're unlikely to need it.

## The minimal change at each rung

The point of the ladder is that each step adds **one concept** to what you
already know. Nothing rewinds.

### 1 — Single agent

```python
agent = Agent("claude-opus-4-7")
print(agent("Hello").text())
```

The string `"claude-opus-4-7"` is sugar for
`Agent(engine=LLMEngine("claude-opus-4-7"))`. Call the agent directly to
run it; `.text()` extracts the string payload from the returned `Envelope`.

### 2 — Agent with tools

```python
agent = Agent("claude-opus-4-7", tools=[get_weather])
print(agent("What's the weather in Paris?").text())
```

The agent decides when to call the tool. You added a list element. The
function's signature, type hints, and docstring become the tool schema —
no JSON to write.

### 3 — Structured output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullets: list[str]

agent = Agent("claude-opus-4-7", output=Summary)
result = agent("Summarise the news")
print(result.payload.headline)   # read .payload, not .text(), with output=
```

You added an `output=` argument. The framework validates the payload
against the model and re-prompts on validation errors.

### 4 — Sequential chain

```python
researcher = Agent("claude-opus-4-7", name="research", tools=[web_search])
writer     = Agent("claude-opus-4-7", name="write")
pipeline   = Agent.chain(researcher, writer)

print(pipeline("Topic: AI agents in 2026").text())
```

Two agents, output of one feeds the next. No new primitive — `chain`
returns an `Agent`, so you can chain chains.

### 5 — Parallel fan-out

```python
multi = Agent.parallel(researcher_a, researcher_b, researcher_c)
envelopes = multi("Same task for everyone")   # returns list[Envelope]
```

Three agents run concurrently against the same input. The result is the
list of their envelopes — `Agent.parallel` returns a deterministic
fan-out, not a different kind of agent.

### 6 — Agent as tool

```python
supervisor = Agent(
    "claude-opus-4-7",
    tools=[researcher],   # researcher's name= becomes the tool name
)
```

The supervisor decides when to delegate to the researcher. This is the
hierarchical / sub-agent pattern, expressed as a tool list. Use
`researcher.as_tool("alias")` only when you need a surface name different
from the agent's `name=`.

### 7 — Deterministic Plan

```python
from lazybridge import Plan, Step

pipeline = Agent(
    engine=Plan(
        Step("research"),
        Step("write"),
    ),
    tools=[researcher, writer],
)

print(pipeline("Topic: AI agents in 2026").text())
```

When you want **the LLM to stop deciding the order**, swap the engine for
`Plan`. The agent's interface stays identical: same `agent(task)` call,
same `Envelope` back. Steps reference sub-agents by `name=` (the same
name they have in `tools=[...]`); the plan is validated at construction
so broken references fail fast, before any LLM call.

### 8 — Parallel band + routing

```python
from lazybridge import Plan, Step

pipeline = Agent(
    engine=Plan(
        Step("triage", routes_by="category"),
        Step("legal",     parallel=True),
        Step("technical", parallel=True),
        Step("reply"),
    ),
    tools=[triage, legal, technical, write_reply],
)
```

A parallel band runs concurrently; a router dispatches based on a
structured field of the previous step's payload.

### 9 — Checkpoint + resume

```python
from lazybridge import Plan, Step, Store

store    = Store(db="runs.db")
pipeline = Agent(
    engine=Plan(*steps, store=store, checkpoint_key="ticket-42"),
    tools=[...],
)
# crash...
resumed = Agent(
    engine=Plan(*steps, store=store, checkpoint_key="ticket-42", resume=True),
    tools=[...],
)
```

Pass a persistent `Store` and a key. After a crash, build the same plan
with `resume=True` and call it again — execution picks up at the failed
step.

### 10 — Human-in-the-loop

```python
from lazybridge import Plan, Step
from lazybridge.ext.hil import human_agent

approval = human_agent(timeout=300, name="approve")
pipeline = Agent(
    engine=Plan(
        Step("draft"),
        Step("approve"),
        Step("send"),
    ),
    tools=[draft, approval, send],
)
```

A human approval is just another agent. Drop it into the plan's
`tools=[...]` and reference its name from a `Step`; the plan halts and
waits when that step runs.

### 11 — Observability

```python
from lazybridge import Agent, Session, JsonFileExporter

session = Session(exporters=[JsonFileExporter("events.jsonl")])
agent   = Agent("claude-opus-4-7", session=session)
```

Add a `Session` once at the top. Every nested agent, tool call, and step
emits events to it. The OpenTelemetry exporter is one extra import away.

### 12 — Custom engine / provider

When you need an engine LazyBridge doesn't ship, implement `BaseEngine`.
When you need a provider LazyBridge doesn't ship, implement `BaseProvider`.
Both are stable extension points.

## The runnable mapping

The repository ships nine example files — one per rung you're likely to
care about. They will become recipe pages in Phase 4 of the documentation;
for now they are runnable from the command line.

| Rung | Example file |
|---|---|
| 2 | `examples/langgraph/01_react_agent_weather.py` |
| 4 | `examples/crewai/02_research_and_report.py` |
| 6 | `examples/langgraph/02_supervisor_research_math.py` |
| 7 | `examples/patterns/plan_tool.py` |
| 7-8 | `examples/patterns/agent_builds_plan.py` |
| 7-8 | `examples/patterns/dynamic_planner.py` |
| 8 | `examples/parallel_report_pipeline.py` |
| 8-11 | `examples/daily_news_report.py` |
| 11 | `examples/viz_demo.py` |

## The anti-pattern worth naming

The biggest mistake new users make is **starting at rung 7**.

Reach for `Plan` when you actually need determinism, validation, parallel
bands, or checkpoints. Reach for sub-agents when a sub-agent has a clear
distinct responsibility. **Don't reach for either because they look more
serious.** A single `Agent` with three tools is often the right shape for
a task that looks complex on the whiteboard.

The framework rewards picking the lowest rung that solves your problem.

## See also

- [Mental model](mental-model.md) — the Engine + Tools + State
  decomposition that all twelve rungs share.
- [Everything is a tool](everything-is-a-tool.md) — the composition
  rule that makes rungs 4-6 a one-line change.
- [Quickstart](../quickstart.md) — rungs 1 and 2, end to end.
