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
sess.store.write("ai_news", researcher._last_output, agent_id=researcher.id)

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

## Concurrent agents with gather()

Run multiple agents at the same time:

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
    # results[0], results[1], results[2] are CompletionResponse objects
    for r in results:
        print(r.content[:200])

asyncio.run(run())
```

All three run concurrently. `gather()` waits for all to finish and returns results in the same order.

---

## Exposing the whole pipeline as a tool

A pipeline (session) can itself be used as a tool by an external orchestrator:

```python
sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)
analyst    = LazyAgent("openai",    name="analyst",    session=sess)

# researcher drives the pipeline; analyst reads researcher's output separately
pipeline_tool = sess.as_tool(
    "research_pipeline",
    "Runs the full research pipeline and returns a summary",
    entry_agent=researcher,
)

# External orchestrator
master = LazyAgent("anthropic")
master.loop("Analyse these 3 topics: fusion energy, quantum computing, biotech", tools=[pipeline_tool])
```

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
sess.store.write("phase1", researcher._last_output)
# process exits — data saved to project.db

# Run 2 — pick up where you left off
sess2 = LazySession(db="project.db")
phase1_data = sess2.store.read("phase1")
writer = LazyAgent("openai", session=sess2)
ctx = LazyContext.from_store(sess2.store, keys=["phase1"])
writer.chat("continue from this research", context=ctx)
```
