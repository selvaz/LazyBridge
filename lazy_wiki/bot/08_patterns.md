# Pipeline Patterns — Complete Runnable Code

## Pattern A — Hierarchy (Orchestrator + Sub-agents)

An orchestrator agent calls sub-agents as tools via `loop()`. Sub-agents are wrapped with `as_tool()` or `LazyTool.from_agent()`.

**Communication**: sub-agent return value → tool result → orchestrator conversation.
**State**: each agent has its own conversation history; no shared session needed.

```python
from lazybridge import LazyAgent, LazyTool

# Sub-agents
researcher = LazyAgent(
    "anthropic",
    name="researcher",
    description="Searches the web and returns factual summaries.",
    system="You are a research assistant. Always cite sources.",
)

analyst = LazyAgent(
    "openai",
    name="analyst",
    description="Analyses data and identifies trends.",
    system="You are a data analyst. Be concise and quantitative.",
)

# Wrap as tools
research_tool = researcher.as_tool()
analysis_tool = analyst.as_tool()

# Orchestrator drives everything
orchestrator = LazyAgent(
    "anthropic",
    system="You coordinate research and analysis tasks.",
)

result = orchestrator.loop(
    "Prepare a report on the current state of open-source LLMs.",
    tools=[research_tool, analysis_tool],
    max_steps=10,
)

print(result.content)
```

**Async version:**
```python
import asyncio

async def main():
    result = await orchestrator.aloop(
        "Prepare a report on open-source LLMs.",
        tools=[research_tool, analysis_tool],
    )
    print(result.content)

asyncio.run(main())
```

**With on_event callback (step-level observability):**
```python
def log_event(event: str, payload):
    if event == "tool_call":
        print(f"  → calling {payload.name}({payload.arguments})")
    elif event == "done":
        print(f"Done in {payload.usage.output_tokens} output tokens")

result = orchestrator.loop(
    "Prepare the report",
    tools=[research_tool, analysis_tool],
    on_event=log_event,
)
```

---

## Pattern B — Parallel (Concurrent Agents, Shared Session)

Multiple agents run concurrently within the same session. Results are collected by `gather()`.

**Communication**: each agent works independently; side effects via `LazyStore`.
**Use case**: fan-out research, parallel processing of different topics.

```python
import asyncio
from lazybridge import LazyAgent, LazySession, LazyContext

sess = LazySession(tracking="verbose")

agent_tech   = LazyAgent("anthropic", name="tech_researcher",   session=sess)
agent_market = LazyAgent("openai",    name="market_researcher", session=sess)
agent_legal  = LazyAgent("anthropic", name="legal_researcher",  session=sess)

async def main():
    results = await sess.gather(
        agent_tech.aloop("Summarise the latest advances in transformer architectures."),
        agent_market.aloop("Summarise the latest AI market trends and funding rounds."),
        agent_legal.aloop("Summarise recent AI regulation developments in the EU and US."),
    )

    # Each result is a CompletionResponse
    tech_result, market_result, legal_result = results

    # Store for cross-agent use
    sess.store.write("tech",   tech_result.content,   agent_id=agent_tech.id)
    sess.store.write("market", market_result.content, agent_id=agent_market.id)
    sess.store.write("legal",  legal_result.content,  agent_id=agent_legal.id)

    # Combine into a final report
    ctx = (
        LazyContext.from_store(sess.store, keys=["tech"])
        + LazyContext.from_store(sess.store, keys=["market"])
        + LazyContext.from_store(sess.store, keys=["legal"])
    )
    editor = LazyAgent("anthropic", name="editor", context=ctx, session=sess)
    report = editor.chat("Write an executive summary combining all three sections.")
    print(report.content)

asyncio.run(main())
```

---

## Pattern C — Network (Cross-loop, No Direct References)

Agents communicate via `LazyStore` (state) and `LazyContext.from_store()` (context injection). Agents don't need to know about each other.

**Use case**: long-running pipelines, agents in separate loops, decoupled modules.

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore(db="pipeline.db")   # persistent SQLite

# --- Loop 1: Data Collection ---
collector = LazyAgent("anthropic", name="collector")
result = collector.loop("collect the top 5 AI papers published this week")
store.write("papers", result.content, agent_id=collector.id)

# --- Loop 2: Analysis (no reference to collector) ---
ctx = LazyContext.from_store(store, keys=["papers"])
analyst = LazyAgent("openai", name="analyst", context=ctx)
analysis = analyst.chat("Identify the 3 most impactful findings from these papers.")
store.write("analysis", analysis.content, agent_id=analyst.id)

# --- Loop 3: Report Writing (no reference to analyst or collector) ---
ctx2 = LazyContext.from_store(store, keys=["papers", "analysis"])
writer = LazyAgent("anthropic", name="writer", context=ctx2)
report = writer.chat("Write a professional newsletter section from this material.")
print(report.content)
```

---

## Pattern D — Router (Conditional Branching)

A `LazyRouter` inspects a result and routes to one of several agents.

```python
from lazybridge import LazyAgent, LazyRouter, LazySession

sess = LazySession()
drafter  = LazyAgent("anthropic", name="drafter",  session=sess)
reviewer = LazyAgent("openai",    name="reviewer", session=sess)
publisher = LazyAgent("anthropic", name="publisher", session=sess)

router = LazyRouter(
    condition=lambda r: "publisher" if "APPROVED" in r.upper() else "reviewer",
    routes={"publisher": publisher, "reviewer": reviewer},
    name="approval_gate",
    default="reviewer",
)

# Pipeline
draft = drafter.chat("Write a short blog post about AI safety.")
next_agent = router.route(draft.content)
result = next_agent.chat("Process this content: " + draft.content)
print(result.content)
```

---

## Pattern E — Pipeline as Tool (Nested Orchestration)

A full pipeline (LazySession) is exposed as a single tool to an external orchestrator.

```python
from lazybridge import LazyAgent, LazySession

# Inner pipeline
inner_sess  = LazySession()
researcher  = LazyAgent("anthropic", name="researcher",  session=inner_sess)
summariser  = LazyAgent("openai",    name="summariser",  session=inner_sess)

def run_pipeline(task: str) -> str:
    res = researcher.loop(task)
    summariser_ctx = LazyContext.from_agent(researcher)
    summary = summariser.chat("summarise", context=summariser_ctx)
    return summary.content

# Expose via LazyTool.from_function for full control:
from lazybridge import LazyTool
pipeline_tool = LazyTool.from_function(
    run_pipeline,
    name="research_and_summarise",
    description="Researches a topic and returns a summary.",
)

# Or via sess.as_tool (entry_agent drives the rest):
pipeline_tool2 = inner_sess.as_tool(
    "research_pipeline",
    "Run full research pipeline",
    entry_agent=researcher,
)

# External orchestrator
orchestrator = LazyAgent("anthropic")
result = orchestrator.loop(
    "Prepare reports on: 1) quantum computing 2) fusion energy 3) space debris",
    tools=[pipeline_tool],
    max_steps=12,
)
print(result.content)
```
