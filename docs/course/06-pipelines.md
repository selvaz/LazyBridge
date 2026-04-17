# Module 6: Multi-Agent Pipelines

Build complex workflows by chaining agents in sequence, running them in parallel, or composing pipelines as tools.

## Sessions — shared state for agents

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess,
                       tools=[web_search])  # tools bound to agent
writer = LazyAgent("openai", name="writer", session=sess)  # no tools
```

A session provides:

- **Store** — shared key-value blackboard (`sess.store`)
- **Events** — tracking/logging (`sess.events`)
- **Graph** — pipeline topology for visualization (`sess.graph`)

**Key pattern:** Bind tools at the agent level (not the session or pipeline level). Each agent carries its own tools. When the pipeline runs, each agent uses its bound tools automatically — the orchestrator doesn't need to know or manage them.

## Chain mode — sequential pipeline

Agents run one after another. Each agent receives the previous agent's output as context:

```python
from lazybridge import LazyAgent, LazySession, LazyTool

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer = LazyAgent("openai", name="writer", session=sess)
editor = LazyAgent("anthropic", name="editor", session=sess)

pipeline = sess.as_tool("pipeline", "Research, write, edit", mode="chain")
result = pipeline.run({"task": "Write an article about AI safety"})
print(result)
```

**Flow:** task -> researcher -> (output as context to) writer -> (output as context to) editor -> final result

## Parallel mode — concurrent execution

All agents process the same task simultaneously:

```python
sess = LazySession()
analyst_a = LazyAgent("anthropic", name="market_analyst", session=sess)
analyst_b = LazyAgent("openai", name="tech_analyst", session=sess)
analyst_c = LazyAgent("google", name="risk_analyst", session=sess)

panel = sess.as_tool("analysis", "Multi-perspective analysis", mode="parallel")
result = panel.run({"task": "Analyze the impact of AI on healthcare"})
print(result)
# [market_analyst]
# ...market perspective...
# [tech_analyst]
# ...tech perspective...
# [risk_analyst]
# ...risk perspective...
```

### Combiners

Control how parallel results are merged:

```python
# concat (default) — all outputs with agent name headers
panel = sess.as_tool("p", "d", mode="parallel", combiner="concat")

# last — only the last agent's output
panel = sess.as_tool("p", "d", mode="parallel", combiner="last")
```

## LazyTool.chain() and LazyTool.parallel()

For pipelines outside a session, use the class methods directly:

```python
from lazybridge import LazyAgent, LazyTool

researcher = LazyAgent("anthropic", name="researcher")
writer = LazyAgent("openai", name="writer")

# Chain
pipeline = LazyTool.chain(researcher, writer, name="pipe", description="Research then write")
result = pipeline.run({"task": "Explain quantum computing"})

# Parallel
panel = LazyTool.parallel(researcher, writer, name="panel", description="Multi-view")
result = panel.run({"task": "Analyze AI trends"})
```

## Mixing agents and tools in chains

Chains can include both agents and function tools:

```python
def fetch_data(task: str) -> str:
    """Fetch raw data from an API."""
    return '{"temperature": 22, "humidity": 65}'

data_tool = LazyTool.from_function(fetch_data)
analyst = LazyAgent("anthropic", name="analyst")

# Tool output becomes the next agent's task
pipeline = LazyTool.chain(data_tool, analyst, name="pipe", description="Fetch then analyze")
result = pipeline.run({"task": "Get weather data and analyze it"})
```

## Checkpoint & resume

Persist pipeline progress so crashed runs can resume:

```python
from lazybridge import LazyStore

store = LazyStore(db="pipeline.db")  # SQLite-backed

pipeline = LazyTool.chain(
    researcher, writer, editor,
    name="article",
    description="Full article pipeline",
    store=store,
    chain_id="article-v1",
)

# Run 1: crashes after step 2
pipeline.run({"task": "Write about fusion energy"})

# Run 2: resumes from step 2 automatically
pipeline.run({"task": "Write about fusion energy"})
```

### Concurrent runs with run_id

Isolate checkpoints for parallel invocations:

```python
# Worker 1
pipeline.run({"task": "topic A"})  # uses default checkpoint key

# Worker 2 — needs its own checkpoint lane
pipeline2 = LazyTool.chain(
    researcher, writer,
    name="pipe", description="d",
    store=store, chain_id="pipe", run_id="worker-2",
)
pipeline2.run({"task": "topic B"})  # independent checkpoint
```

## Pipeline as a tool for an orchestrator

The most powerful pattern — expose a pipeline as a tool for a higher-level agent:

```python
# Build specialized pipelines
research_pipeline = LazyTool.chain(
    data_fetcher, analyst,
    name="research", description="Fetch data and analyze it",
)

writing_pipeline = LazyTool.chain(
    researcher, writer, editor,
    name="write", description="Research, write, and edit an article",
)

# Orchestrator decides which pipeline to use
orchestrator = LazyAgent("anthropic", system="You coordinate research and writing tasks.")
resp = orchestrator.loop(
    "First research AI trends, then write an article about the most interesting one",
    tools=[research_pipeline, writing_pipeline],
)
```

## Routing — conditional branching

Use `LazyRouter` for if/else logic in pipelines:

```python
from lazybridge import LazyRouter

router = LazyRouter(
    condition=lambda result: "technical" if "code" in result.lower() else "general",
    routes={
        "technical": LazyAgent("anthropic", system="You are a technical writer."),
        "general": LazyAgent("openai", system="You are a general writer."),
    },
    name="writer_selector",
)

result = researcher.chat("Find info about Python decorators")
next_agent = router.route(result.content)
final = next_agent.chat(f"Write about: {result.content}")
```

## Async pipelines

```python
import asyncio

async def main():
    sess = LazySession()
    a = LazyAgent("anthropic", name="a", session=sess)
    b = LazyAgent("openai", name="b", session=sess)

    # Concurrent execution
    results = await sess.gather(
        a.aloop("Research topic X"),
        b.aloop("Research topic Y"),
    )
    print(results)

asyncio.run(main())
```

---

## Exercise

1. Build a 3-agent chain: researcher -> writer -> translator (translates to Italian)
2. Build a parallel panel with 3 agents analyzing the same topic from different angles
3. Add checkpoint persistence with a SQLite store and verify resume works
4. Create a pipeline-as-tool and give it to an orchestrator agent

**Next:** [Module 7: Streaming](07-streaming.md) — real-time token-by-token output.
