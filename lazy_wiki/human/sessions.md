# LazySession — Multi-agent Pipelines

A `LazySession` is the shared container for a multi-agent pipeline. Create one when you have more than one agent that needs to share state, tracking, or a graph.

---

## When do you need a session?

| Scenario | Session needed? |
|----------|----------------|
| Single agent, one-off question | No |
| Single agent with tools | No |
| Two agents sharing results | Yes |
| Agents running concurrently | Yes |
| Persistent state across runs | Yes |
| You want event tracking | Yes |
| GUI pipeline visualization | Yes |

---

## Creating a session

```python
from lazybridge import LazySession

# In-memory (default) — data lost when process exits
sess = LazySession()

# Persistent — SQLite file stores events and state
sess = LazySession(db="my_pipeline.db")

# Control tracking verbosity
sess = LazySession(tracking="off")      # no tracking
sess = LazySession(tracking="basic")    # default: all events except stream chunks
sess = LazySession(tracking="verbose")  # everything including stream chunks
```

Attach agents to the session via `session=`:

```python
from lazybridge import LazyAgent

researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)
# Both agents now share sess.store, sess.events, and appear in sess.graph
```

---

## Shared store

`sess.store` is a key-value blackboard. Any agent in the session can write to and read from it.

```python
# Agent writes results
researcher.loop("find top AI news")
sess.store.write("ai_news", researcher.result, agent_id=researcher.id)

# Another agent reads without needing a reference to researcher
from lazybridge import LazyContext
ctx = LazyContext.from_store(sess.store, keys=["ai_news"])
writer.chat("write a newsletter section", context=ctx)

# Read everything
print(sess.store.read_all())

# Read only what a specific agent wrote
print(sess.store.read_by_agent(researcher.id))
```

See [context.md](context.md) for how `LazyContext.from_store` works.

---

## Event tracking

Every LLM call, tool invocation, and loop step is automatically logged.

```python
from lazybridge import Event

# All events
all_events = sess.events.get()

# Filter by event type
tool_calls = sess.events.get(event_type=Event.TOOL_CALL)
responses  = sess.events.get(event_type=Event.MODEL_RESPONSE)

# Filter by agent
writer_events = sess.events.get(agent_id=writer.id)

# Both filters + limit
recent = sess.events.get(agent_id=researcher.id, event_type=Event.TOOL_RESULT, limit=10)

# Each event is a dict:
for ev in all_events:
    print(ev["timestamp"], ev["agent_name"], ev["event_type"], ev["data"])
```

Event types you'll see at `BASIC` level:
- `MODEL_REQUEST` — before each LLM call
- `MODEL_RESPONSE` — after each LLM call (includes token usage)
- `TOOL_CALL` — model requested a tool
- `TOOL_RESULT` — tool returned successfully
- `TOOL_ERROR` — tool raised an exception
- `LOOP_STEP` — each iteration of `loop()`

---

## Exposing the pipeline as a tool

The cleanest way to run agents together — as a parallel fan-out or a sequential chain — is `sess.as_tool()`. This composes the entire pipeline into a single callable that an orchestrator can invoke by name, with no asyncio boilerplate and no manual context wiring.

### mode="parallel" — all agents receive the same task, results combined

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession()

gather_news = sess.as_tool(
    "gather_global_news",
    "Simultaneously gather AI news from the US, Europe, and Asia. Returns combined results.",
    mode="parallel",
    participants=[
        LazyAgent("anthropic", name="us_news",   session=sess),
        LazyAgent("openai",    name="eu_news",   session=sess),
        LazyAgent("google",    name="asia_news", session=sess),
    ],
    combiner="concat",   # join outputs with newlines (default)
)

editor = LazyAgent("anthropic", name="editor", session=sess)
editor.loop(
    "Gather today's AI news from the US, Europe, and Asia, then write a 400-word global digest.",
    tools=[gather_news],
)
```

All three agents run concurrently. The combined output is returned to the orchestrator as a single string.

### mode="chain" — each agent's output flows into the next

```python
sess = LazySession()
pipeline_tool = sess.as_tool(
    "research_pipeline",
    "Researches a topic, then produces an analysis. Returns the analyst's report.",
    mode="chain",
    participants=[
        LazyAgent("anthropic", name="researcher", session=sess),
        LazyAgent("openai",    name="analyst",    session=sess),
    ],
)

master = LazyAgent("anthropic")
master.loop("Analyse these 3 topics: fusion energy, quantum computing, biotech", tools=[pipeline_tool])
```

No `LazyContext` wiring needed — the chain passes each agent's output to the next automatically.

---

## Lower-level concurrency with gather()

Use `gather()` directly when you need the raw `CompletionResponse` from each agent rather than a combined string — for example, to inspect token usage or tool calls per agent before merging results yourself.

```python
import asyncio
from lazybridge import LazyAgent, LazySession

sess = LazySession()
agent_a = LazyAgent("anthropic", name="news_a", session=sess)
agent_b = LazyAgent("openai",    name="news_b", session=sess)
agent_c = LazyAgent("google",    name="news_c", session=sess)

async def run():
    results = await sess.gather(
        agent_a.aloop("Summarise AI news from the US this week"),
        agent_b.aloop("Summarise AI news from Europe this week"),
        agent_c.aloop("Summarise AI news from Asia this week"),
    )
    # results[i] are CompletionResponse objects — full access to usage, tool_calls, etc.
    for r in results:
        print(r.content[:200])
        print(f"  tokens: {r.usage.input_tokens}in / {r.usage.output_tokens}out")

asyncio.run(run())
```

For most cases where you just want concurrent agents and a combined result, `sess.as_tool(mode="parallel")` is simpler.

---

## Graph serialization (for GUI)

The session's pipeline topology is automatically captured and can be exported:

```python
# Export as JSON (for a GUI to load)
json_str = sess.to_json()
print(json_str)

# Save to file
sess.graph.save("pipeline.json")
sess.graph.save("pipeline.yaml")   # requires pip install pyyaml
```

---

## Persistent sessions

Use `db=` to persist both the event log and the store across runs:

```python
# Run 1
sess = LazySession(db="project.db")
researcher = LazyAgent("anthropic", session=sess)
researcher.loop("research phase 1")
sess.store.write("phase1", researcher.result)
# process exits — data saved to project.db

# Run 2 — pick up where you left off
from lazybridge import LazyAgent, LazySession, LazyContext  # agents must be reconstructed each run

sess2 = LazySession(db="project.db")
writer = LazyAgent("openai", session=sess2)
ctx = LazyContext.from_store(sess2.store, keys=["phase1"])
writer.chat("continue from this research", context=ctx)
```
