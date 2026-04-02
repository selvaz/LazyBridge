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

Shared key-value blackboard. See `05_lazystore.md` for the full API. Key pattern: one agent writes results, another reads them via `LazyContext.from_store()` — neither agent needs to know about the other directly.

```python
from lazybridge import LazyAgent, LazySession, LazyContext

sess = LazySession()
agent_a = LazyAgent("anthropic", name="researcher", session=sess)
agent_b = LazyAgent("openai",    name="writer",     session=sess)

agent_a.loop("research the topic of quantum computing")

# Write agent_a's output to the store explicitly
sess.store.write("research", agent_a._last_output, agent_id=agent_a.id)

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

## 6. `gather()` — concurrent execution

```python
async def gather(self, *coros: Awaitable) -> list[Any]:
```

Thin wrapper over `asyncio.gather()`. Runs multiple agent coroutines in parallel; results are returned in the same order as the arguments.

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
    # results[0] = agent_a's CompletionResponse
    # results[1] = agent_b's CompletionResponse
    print(sess.store.read_all())

asyncio.run(run())
```

Use `gather()` for independent parallel agents. Agents that depend on each other's output must be sequenced: `await agent_a.aloop(...)` then `await agent_b.aloop(...)`.

---

## 7. `as_tool()` — expose the pipeline to an external orchestrator

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

### `mode="chain"` — agents run sequentially, each receives the previous output

Participants execute in order. The first receives the original task; each subsequent participant receives the previous one's output as its context. The final participant's output is returned.

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

## 8. Serialization

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
