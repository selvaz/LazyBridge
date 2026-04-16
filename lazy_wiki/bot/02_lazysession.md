# LazySession — Complete Reference

## 1. Overview

`LazySession` is the shared container for a multi-agent pipeline. It holds:
- `store: LazyStore` — shared blackboard; any agent can read/write keyed values
- `events: EventLog` — SQLite-backed event tracking with configurable verbosity
- `graph: GraphSchema` — auto-built directed graph of registered agents and tools

Agents register themselves in `sess.graph` at construction time (when `session=sess` is passed to `LazyAgent`).

---

## 2. Constructor

```python
LazySession(
    *,
    db: str | None = None,
    tracking: TrackLevel | str = TrackLevel.BASIC,
    console: bool = False,
)
```

```python
from lazybridge import LazySession

sess = LazySession()                                          # in-memory, BASIC tracking
sess = LazySession(db="pipeline.db")                          # SQLite-backed store + events
sess = LazySession(tracking="verbose")                        # in-memory, VERBOSE tracking
sess = LazySession(db="run.db", tracking="off")               # SQLite, no tracking
sess = LazySession(tracking="basic", console=True)            # print events to stdout in real-time
```

`db` is a file path. Both `LazyStore` and `EventLog` use the same SQLite file when `db` is set. `tracking` accepts `TrackLevel` enum values or their string equivalents (`"off"`, `"basic"`, `"verbose"`, `"full"`).

`console=True` prints all tracked events to stdout in real-time as agents run — useful for debugging pipelines without opening the DB. When `verbose=True` is passed to a `LazyAgent` that belongs to this session, `console` is automatically enabled on the session.

---

## 3. `sess.store: LazyStore`

Shared key-value blackboard. See `05_lazystore.md` for the full API.

**When to use `LazyStore`:** prefer it when agents are **intentionally decoupled** — running at different times, in separate processes, or when any step may be skipped. For tightly coupled sequential agents where every output feeds the next, `sess.as_tool(mode="chain")` is simpler — no store writes, no explicit context wiring needed.

```python
from lazybridge import LazyAgent, LazySession, LazyContext

sess = LazySession()
agent_a = LazyAgent("anthropic", name="researcher", session=sess)
agent_b = LazyAgent("openai",    name="writer",     session=sess)

agent_a.loop("research the topic of quantum computing")

# Use agent.result (canonical accessor), not _last_output
sess.store.write("research", agent_a.result, agent_id=agent_a.id)

# agent_b reads it without knowing about agent_a
ctx = LazyContext.from_store(sess.store, keys=["research"])
agent_b.chat("write a summary", context=ctx)
```

---

## 4. `sess.events: EventLog` — tracking

### `get()` — query events

```python
EventLog.get(
    *,
    agent_id: str | None = None,
    event_type: str | None = None,
    limit: int = 200,
) -> list[dict]
```

```python
from lazybridge import LazyAgent, LazySession
from lazybridge.lazy_session import Event

sess = LazySession()
agent_a = LazyAgent("anthropic", name="a", session=sess)

# All events in this session
all_events = sess.events.get()

# Filter by agent
sess.events.get(agent_id=agent_a.id)

# Filter by event type
sess.events.get(event_type=Event.TOOL_CALL)

# Combined filter with limit
sess.events.get(agent_id=agent_a.id, event_type=Event.MODEL_RESPONSE, limit=10)
```

Each event dict has keys: `timestamp`, `agent_id`, `agent_name`, `event_type`, `data`.

### `TrackLevel` values

| Value | What is logged |
|---|---|
| `TrackLevel.OFF` | Nothing — log calls are no-ops |
| `TrackLevel.BASIC` | All events except `MESSAGES`, `SYSTEM_CONTEXT`, `STREAM_CHUNK` |
| `TrackLevel.VERBOSE` | Everything, including high-volume stream chunks |
| `TrackLevel.FULL` | Synonym for `VERBOSE` |

### `Event` types

Logged at `BASIC` and above:

```python
Event.MODEL_REQUEST    # before each LLM call
Event.MODEL_RESPONSE   # after each LLM call (includes stop_reason, token counts)
Event.TOOL_CALL        # each tool invocation (name + arguments)
Event.TOOL_RESULT      # tool return value (truncated to 500 chars in data)
Event.TOOL_ERROR       # exception from tool execution
Event.AGENT_START      # reserved
Event.AGENT_FINISH     # reserved
Event.LOOP_STEP        # reserved
```

Logged at `VERBOSE` only (high-volume):

```python
Event.MESSAGES         # full message list before each call
Event.SYSTEM_CONTEXT   # resolved system prompt
Event.STREAM_CHUNK     # each streaming delta chunk
```

---

## 5. `sess.graph: GraphSchema` — auto-built topology

The graph is populated automatically as agents are constructed with `session=sess`. No manual registration calls are needed.

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession()
a = LazyAgent("anthropic", name="researcher", session=sess)
b = LazyAgent("openai",    name="analyst",    session=sess)

# Both nodes are already registered
print(sess.graph.to_json())
# {
#   "session_id": "...",
#   "nodes": [
#     {"id": "...", "name": "researcher", "provider": "anthropic", "model": "...", ...},
#     {"id": "...", "name": "analyst",    "provider": "openai",    "model": "...", ...}
#   ],
#   "edges": []
# }
```

Edges are added manually via `sess.graph.add_edge(from_id, to_id, kind=EdgeType.TOOL)`. See `GraphSchema` in `00_quickref.md` for the full graph API.

---

## 6. `as_tool()` — expose the pipeline to an external orchestrator — CANONICAL

```python
def as_tool(
    self,
    name: str,
    description: str,
    *,
    mode: str | None = None,
    participants: list[LazyAgent | LazyTool] | None = None,
    combiner: str = "concat",
    entry_agent: LazyAgent | None = None,
    guidance: str | None = None,
) -> LazyTool:
```

Wraps one or more agents (and/or nested `LazyTool`s) as a single `LazyTool`. The tool schema is always `{"task": str}`. The orchestrator passes a task string; the participants receive it.

**Implementation note:** `as_tool(mode="parallel")` and `as_tool(mode="chain")` are thin wrappers over `LazyTool.parallel()` and `LazyTool.chain()` respectively — semantically identical. Use `LazyTool.parallel()` / `LazyTool.chain()` directly when you don't have a session. The returned tool has `_is_pipeline_tool = True`; `save()` raises `ValueError` on it.

**Cross-session validation:** if participants are bound to different sessions, `as_tool()` raises `ValueError` at creation time. This also covers `LazyTool.from_agent()` tools — the inner agent's session is checked, not just direct `LazyAgent` participants.

### `mode="parallel"` — all agents receive the same task concurrently

All participants run in parallel on the same input task. Their outputs are combined (default: concatenated with agent-name headers) and returned as a single string.

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession(tracking="basic", console=True)
LazyAgent("anthropic", name="us_analyst",     session=sess, system="US market analyst")
LazyAgent("openai",    name="europe_analyst",  session=sess, system="European market analyst")
LazyAgent("google",    name="asia_analyst",    session=sess, system="Asian market analyst")

news_tool = sess.as_tool(
    "global_market_news",
    "Parallel market analysis across US, Europe, and Asia",
    mode="parallel",
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop("Prepare a global market brief", tools=[news_tool])
```

`participants` defaults to all agents registered in the session (in registration order). Pass an explicit list to control order or include `LazyTool` instances.

`combiner="concat"` (default) joins outputs as:
```
[us_analyst]
<output>

[europe_analyst]
<output>
```
`combiner="last"` returns only the last participant's output.

### `mode="chain"` — sequential pipeline with explicit handoff semantics

Participants execute in order. Each step receives the result of the previous step, but **the handoff mechanism differs depending on whether the previous step was an agent or a tool**. Understanding this contract is important when mixing agents and tools in the same chain.

---

#### Chain handoff contract

**Input routing — what the current step receives:**

| Previous step | Current step receives |
|---|---|
| Nothing (first step) | Original task string as message |
| `LazyAgent` | Original task string as message **+ previous agent's output injected via `LazyContext`** |
| `LazyTool` | Previous tool's output **as the new task string** (replaces original task) |

This asymmetry is intentional and maps to real semantic intent:

- When an **agent** precedes another agent, its output is *contextual and interpretive* — it belongs in the system prompt alongside the original goal. The next agent still "knows what it needs to do" (original task) while also "knowing what was found" (context).
- When a **tool** precedes an agent, its output is *raw data to process* — it is the new input, not supplementary context. The original task no longer makes sense as the message; the tool result does.

```
LazyAgent → LazyAgent:
  message  = original task        ("Analyse the EV market")
  context  = previous agent output (injected into system prompt via LazyContext)

LazyTool → LazyAgent:
  message  = tool's return value   ("Market data: EV sales up 40% in Q3...")
  context  = none
```

**Call dispatch — how the current agent is invoked:**

| Agent has | Method called |
|---|---|
| `output_schema` set | `p.json(task, schema)` — structured output + JSON suffix enforcement |
| `tools` or `native_tools` | `p.loop(task)` — tool-calling loop required |
| Neither | `p.chat(task)` — single turn |

**Return value:**

| Last step produced | Chain returns |
|---|---|
| Pydantic object (agent with `output_schema`) | The typed Pydantic object directly |
| Text (agent without schema, or tool) | Plain string |

When the chain is used as a tool inside another agent's `loop()`, the executor serialises Pydantic objects via `model_dump_json()` automatically.

---

#### Basic example — three agents

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession(tracking="basic", console=True)
researcher = LazyAgent("anthropic", name="researcher", session=sess,
                        system="Research analyst. Gather findings.")
analyst    = LazyAgent("openai",    name="analyst",    session=sess,
                        system="Risk analyst. Evaluate the findings.")
writer     = LazyAgent("anthropic", name="writer",     session=sess,
                        system="Report writer. Produce a final report.")

pipeline_tool = sess.as_tool(
    "research_pipeline",
    "Full research → analysis → report pipeline",
    mode="chain",
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop("Analyse the EV market", tools=[pipeline_tool])
```

Flow (all agent → agent steps):
1. `researcher` receives `"Analyse the EV market"` as message, no prior context.
2. `analyst` receives `"Analyse the EV market"` as message + researcher's findings via `LazyContext`.
3. `writer` receives `"Analyse the EV market"` as message + analyst's risk profile via `LazyContext`.

---

#### Typed chain — Pydantic schemas between steps

```python
from pydantic import BaseModel
from lazybridge import LazyAgent, LazySession

class RiskProfile(BaseModel):
    risks: list[str]
    severity: str

class InvestmentReport(BaseModel):
    title: str
    recommendation: str
    risk_summary: str

sess         = LazySession()
risk_analyst = LazyAgent("anthropic", name="risk_analyst",
                          output_schema=RiskProfile, session=sess)
report_writer = LazyAgent("openai",   name="report_writer",
                           output_schema=InvestmentReport, session=sess)

pipeline = sess.as_tool("typed_pipeline", "Risk → Report", mode="chain")
result = pipeline.run({"task": "Analyse the EV market"})

# result is an InvestmentReport instance (typed, not a string)
print(result.title)
print(result.recommendation)
```

`risk_analyst` runs `json(task, RiskProfile)` → produces a `RiskProfile`.
`report_writer` runs `json(task, InvestmentReport)` with `risk_analyst`'s output as context → produces an `InvestmentReport`.
The chain returns the `InvestmentReport` directly — no `json.loads`, no `.parsed`.

---

#### Mixed tool + agent chain

```python
# Tool step followed by agent step — tool output becomes agent's message

market_tool   = market_sess.as_tool("market_research", "...", mode="parallel")
risk_analyst  = LazyAgent("anthropic", output_schema=RiskProfile,  session=analysis_sess)
report_writer = LazyAgent("openai",    output_schema=InvestmentReport, session=analysis_sess)

pipeline = LazySession().as_tool(
    "full_pipeline", "...",
    mode="chain",
    participants=[market_tool, risk_analyst, report_writer],
)
result = pipeline.run({"task": TASK})
```

Flow (tool → agent → agent):
1. `market_tool.run({"task": TASK})` → returns combined parallel research as a string.
2. `risk_analyst` receives the tool's string output as its **message** (not as context). No `LazyContext` injection.
3. `report_writer` receives `TASK` as message + `risk_analyst`'s `RiskProfile` JSON as context.

This is the **only asymmetry in the chain**: a tool's output replaces the task, while an agent's output becomes context. If you want tool output to be treated as context (not task replacement), wrap the tool inside an agent (`agent.as_tool()`) and add a thin pass-through agent step before the next consumer.

---

#### Participant type validation

Chain participants must be either:
- A `LazyAgent` instance (has `.chat()`)
- A `LazyTool` instance (has `.run()`)

Passing any other object raises `TypeError` immediately with a descriptive message.

### Mixing `LazyTool` and `LazyAgent` participants

`participants` can mix `LazyAgent` instances and `LazyTool` instances (including tools created by other sessions). This enables nested pipelines — for example a parallel tier as the first step of a chain:

```python
# Inner session: three parallel researchers
market_sess = LazySession(tracking="basic", console=True)
LazyAgent("google", name="tech_researcher",    session=market_sess,
          native_tools=[NativeTool.WEB_SEARCH], system="Technology analyst")
LazyAgent("google", name="energy_researcher",  session=market_sess,
          native_tools=[NativeTool.WEB_SEARCH], system="Energy analyst")

market_tool = market_sess.as_tool(
    "market_research", "Parallel web research across tech and energy", mode="parallel"
)

# Outer session: chain starts with the parallel tool
analysis_sess = LazySession(tracking="basic", console=True)
risk_analyst  = LazyAgent("anthropic", name="risk_analyst",  session=analysis_sess,
                           output_schema=RiskProfile)
report_writer = LazyAgent("openai",    name="report_writer", session=analysis_sess,
                           output_schema=InvestmentReport)

pipeline = LazySession(tracking="basic", console=True).as_tool(
    "full_pipeline",
    "Parallel research → risk assessment → report",
    mode="chain",
    participants=[market_tool, risk_analyst, report_writer],
)

result = pipeline.run({"task": TASK})
```

### Legacy: `entry_agent=` (backward-compatible)

For single-agent delegation, pass `entry_agent=` without `mode`. This is a thin wrapper over `LazyTool.from_agent(entry_agent, ...)`.

```python
pipeline_tool = sess.as_tool(
    "research_pipeline",
    "Single-agent research pipeline",
    entry_agent=researcher,
)
```

---

## 7. `gather()` — low-level concurrent execution — FALLBACK

**Prefer `sess.as_tool(mode="parallel")` for most fan-out pipelines.** Use `gather()` only when you need the raw `CompletionResponse` from each agent — for example, to inspect per-agent token usage, tool call sequences, or grounding sources before merging results yourself.

```python
async def gather(self, *coros: Awaitable) -> list[Any]:
```

Thin wrapper over `asyncio.gather()`. Results are returned in the same order as arguments.

```python
import asyncio
from lazybridge import LazyAgent, LazySession

sess = LazySession()
agent_a = LazyAgent("anthropic", name="researcher", session=sess)
agent_b = LazyAgent("openai",    name="analyst",    session=sess)

async def run():
    results = await sess.gather(
        agent_a.aloop("research topic X"),
        agent_b.aloop("research topic Y"),
    )
    # results[0] = agent_a's CompletionResponse — full access to .usage, .tool_calls, etc.
    # results[1] = agent_b's CompletionResponse
    print(f"Agent A: {results[0].usage.output_tokens} tokens")
    print(f"Agent B: {results[1].usage.output_tokens} tokens")

asyncio.run(run())
```

If you only need the combined text (not per-agent response objects), `sess.as_tool(mode="parallel")` is canonical — no asyncio, no manual result wiring.

---

## 8. Usage Summary — cost and token tracking

```python
summary = sess.usage_summary()
# Returns:
# {
#     "total": {"input_tokens": 1500, "output_tokens": 800, "cost_usd": 0.023},
#     "by_agent": {
#         "researcher": {"input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.015},
#         "writer":     {"input_tokens": 500,  "output_tokens": 300, "cost_usd": 0.008},
#     },
# }
```

Aggregates token counts and costs from all `model_response` events in the session. Requires `tracking="verbose"` (model_response events are verbose-only).

---

## 9. Exporters — external observability

Register exporters to forward events to external systems:

```python
from lazybridge import LazySession, CallbackExporter, OTelExporter, StructuredLogExporter

# Simple callback
events = []
sess = LazySession(exporters=[CallbackExporter(events.append)])

# Structured JSON logging (stdlib logging, no extra deps)
sess = LazySession(exporters=[StructuredLogExporter()])

# OpenTelemetry spans (requires: pip install lazybridge[otel])
sess = LazySession(exporters=[OTelExporter(service_name="my-pipeline")])
```

Available exporters:
- **`CallbackExporter(fn)`** — wraps any callable
- **`FilteredExporter(inner, event_types={"tool_call", ...})`** — forwards only specified event types
- **`JsonFileExporter("events.jsonl")`** — appends JSON lines to a file
- **`StructuredLogExporter()`** — emits events as structured JSON via Python's logging module
- **`OTelExporter()`** — maps events to OpenTelemetry spans (agent, tool, model spans with token/cost attributes)

Add/remove at runtime:
```python
sess.add_exporter(my_exporter)
sess.remove_exporter(my_exporter)
```

---

## 10. Serialization

```python
# Serialize graph topology to JSON (nodes + edges; live agents are NOT included)
json_str = sess.to_json()
with open("pipeline.json", "w") as f:
    f.write(json_str)

# Restore graph structure from JSON
# Live agents must be re-created separately — reconstruction from JSON is not yet supported
sess2 = LazySession.from_json(json_str)
print(sess2.graph.nodes())   # list of AgentNode descriptors, not LazyAgent instances
```

`to_json()` delegates to `sess.graph.to_json()`. `from_json(text, **kwargs)` is a classmethod that creates a new `LazySession` (forwarding `**kwargs` to the constructor, e.g. `db=`), then replaces its `graph` with the deserialized `GraphSchema` and restores `session_id` from the JSON.

### Resuming from a database

```python
# Resume latest session (default)
sess = LazySession.from_db("pipeline.db")

# Resume a specific session by ID
sess = LazySession.from_db("pipeline.db", session_id="abc-123-...")
```

`from_db(db, *, session_id=None, tracking="basic")`:
- If `session_id` is provided, binds to that specific session — only its events are visible via `events.get()`
- If `session_id` is None, auto-detects the most recent session in the database
- Raises `FileNotFoundError` if the database file doesn't exist
