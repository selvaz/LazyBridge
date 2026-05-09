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
agent = Agent.from_model("claude-sonnet-4-6")
result = await agent.run("Hello")
```

### 2 — Agent with tools

```python
agent = Agent.from_model("claude-sonnet-4-6", tools=[get_weather])
```

The agent decides when to call the tool. You added a list element.

### 3 — Structured output

```python
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullets: list[str]

agent = Agent.from_model("claude-sonnet-4-6", output=Summary)
result = await agent.run("Summarise the news")
print(result.payload.headline)
```

You added an `output=` argument. The framework validates the payload
against the model and re-prompts on validation errors.

### 4 — Sequential chain

```python
researcher = Agent.from_model("claude-sonnet-4-6", tools=[web_search])
writer     = Agent.from_model("claude-sonnet-4-6")
pipeline   = Agent.chain(researcher, writer)
```

Two agents, output of one feeds the next. No new primitive — `chain`
returns an `Agent`, so you can chain chains.

### 5 — Parallel fan-out

```python
multi = Agent.parallel(researcher_a, researcher_b, researcher_c)
```

Three agents run concurrently against the same input. The result is the
list of their envelopes. Same `Agent` shape.

### 6 — Agent as tool

```python
supervisor = Agent.from_model(
    "claude-sonnet-4-6",
    tools=[researcher.as_tool("research", "Look things up online")],
)
```

The supervisor decides when to delegate to the researcher. This is the
hierarchical / sub-agent pattern, expressed as a tool list.

### 7 — Deterministic Plan

```python
from lazybridge import Plan, Step

plan = Plan(
    Step(researcher, name="research"),
    Step(writer,     name="write"),
)
agent = Agent.from_engine(plan)
```

When you want **the LLM to stop deciding the order**, switch the engine
from `LLMEngine` (implicit in `from_model`) to `Plan`. Same `Agent`
interface. The plan is validated at construction — broken references fail
fast, not at runtime.

### 8 — Parallel band + routing

```python
plan = Plan(
    Step(triage,      name="triage", routes_by="category"),
    Step(legal,       name="legal", parallel=True),
    Step(technical,   name="technical", parallel=True),
    Step(write_reply, name="reply"),
)
```

A parallel band runs concurrently; a router dispatches based on a
structured field of the previous step's payload.

### 9 — Checkpoint + resume

```python
plan = Plan(*steps, store=Store(db="runs.db"), checkpoint_key="ticket-42")
# crash...
plan = Plan(*steps, store=Store(db="runs.db"), checkpoint_key="ticket-42", resume=True)
```

Pass a persistent `Store` and a key. After a crash, re-run with
`resume=True` and the plan picks up at the failed step.

### 10 — Human-in-the-loop

```python
from lazybridge import Agent
from lazybridge.ext.hil import human_agent

approval = human_agent(timeout=300)
plan = Plan(
    Step(draft,    name="draft"),
    Step(approval, name="approve"),
    Step(send,     name="send"),
)
```

A human approval is just another step. The plan halts and waits.

### 11 — Observability

```python
from lazybridge import Session, JsonFileExporter

session = Session(exporters=[JsonFileExporter("events.jsonl")])
agent   = Agent.from_model("claude-sonnet-4-6", session=session)
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
