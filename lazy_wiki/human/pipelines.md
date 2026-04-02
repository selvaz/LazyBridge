# Pipeline Walkthroughs

End-to-end pipeline examples. Each example is self-contained and runnable.

---

## Pipeline 1 — Research Report (Hierarchy, Pattern A)

An orchestrator calls a researcher and an analyst as tools. Produces a structured report.

```python
from lazybridge import LazyAgent, LazyTool

# Sub-agents
researcher = LazyAgent(
    "anthropic",
    name="researcher",
    description="Searches for factual information on any topic.",
    system="You are a meticulous researcher. Always provide specific facts and data.",
)

analyst = LazyAgent(
    "openai",
    name="analyst",
    description="Analyses data and draws conclusions.",
    system="You are a data analyst. Be concise. Use numbers when possible.",
)

# Expose sub-agents as tools
research_tool = researcher.as_tool()
analysis_tool = analyst.as_tool()

# Orchestrator coordinates everything
orchestrator = LazyAgent(
    "anthropic",
    system="You coordinate research and analysis tasks. Always produce a final structured report.",
)

result = orchestrator.loop(
    "Prepare a 3-section report on open-source AI model releases in 2024.",
    tools=[research_tool, analysis_tool],
    max_steps=12,
)
print(result.content)
```

---

## Pipeline 2 — News Aggregator (Parallel, Pattern B)

Three agents run concurrently and their results are combined by an editor. The parallel gathering is exposed as a single tool — no asyncio boilerplate, no manual store writes, no context wiring.

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession()

# Parallel tool: all three agents receive the same task and results are concatenated
gather_news = sess.as_tool(
    "gather_global_news",
    "Simultaneously gather AI news from the US, Europe, and Asia",
    mode="parallel",
    participants=[
        LazyAgent("anthropic", name="us_news",   session=sess),
        LazyAgent("openai",    name="eu_news",   session=sess),
        LazyAgent("google",    name="asia_news", session=sess),
    ],
    combiner="concat",
)

editor = LazyAgent("anthropic", name="editor", session=sess)
newsletter = editor.loop(
    "Gather today's AI news from the US, Europe, and Asia, then write a 400-word global digest.",
    tools=[gather_news],
)
print(newsletter.content)
```

---

## Pipeline 3 — Decoupled Analysis (Network, Pattern C)

Three agents that don't know about each other. Communication flows through `LazyStore`.

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore(db="analysis.db")  # persistent

# --- Phase 1: Data Collection ---
collector = LazyAgent("anthropic", name="collector")
collector.loop("List the top 10 Python packages by GitHub stars in 2024.")
store.write("raw_data", collector._last_output, agent_id=collector.id)
print("Phase 1 done:", len(store.read("raw_data")), "chars")

# --- Phase 2: Analysis (no reference to collector) ---
ctx = LazyContext.from_store(store, keys=["raw_data"])
analyst = LazyAgent("openai", name="analyst", context=ctx)
analysis = analyst.chat("Identify the 3 dominant trends from this package data.")
store.write("trends", analysis.content, agent_id=analyst.id)
print("Phase 2 done")

# --- Phase 3: Report (no reference to analyst or collector) ---
ctx2 = (
    LazyContext.from_text("Write for a technical audience. Use markdown headings.")
    + LazyContext.from_store(store, keys=["raw_data", "trends"])
)
writer = LazyAgent("anthropic", name="writer", context=ctx2)
report = writer.chat("Write a comprehensive analysis report from this data.")
print(report.content)
```

---

## Pipeline 4 — Iterative Review Loop (Router)

A drafter and reviewer work in a loop until the content is approved.

```python
from lazybridge import LazyAgent, LazyRouter, LazySession

sess = LazySession(tracking="verbose")

drafter   = LazyAgent("anthropic", name="drafter",   session=sess)
reviewer  = LazyAgent("openai",    name="reviewer",  session=sess)
publisher = LazyAgent("anthropic", name="publisher", session=sess)

router = LazyRouter(
    condition=lambda r: "publish" if "APPROVED" in r.upper() else "revise",
    routes={"publish": publisher, "revise": drafter},
    name="review_gate",
    default="revise",
)

# Initial draft
draft = drafter.chat("Write a 200-word intro to transformer architecture.")

for revision in range(4):   # max 4 revision cycles
    review = reviewer.chat(
        f"Review this text critically. End your response with APPROVED or REJECTED.\n\n{draft.content}"
    )
    print(f"Revision {revision + 1}: {review.content[:80]}...")

    next_agent = router.route(review.content)
    if next_agent is publisher:
        final = publisher.chat(f"Format for publication:\n\n{draft.content}")
        print("\n=== PUBLISHED ===")
        print(final.content)
        break
    else:
        draft = drafter.chat(
            f"Rewrite based on this feedback: {review.content}\n\nOriginal: {draft.content}"
        )
```

---

## Pipeline 5 — Nested Pipelines (Pipeline as Tool)

An outer orchestrator manages two inner pipelines, each exposed as a single tool. Inner pipelines are declared with `as_tool(mode="chain")` — no wrapper functions, no manual context wiring.

```python
from lazybridge import LazyAgent, LazySession

# Inner pipeline A: research → summarise (chain: summariser receives researcher's output)
sess_a = LazySession()
research_tool = sess_a.as_tool(
    "research",
    "Deep-research any topic and return a concise summary.",
    mode="chain",
    participants=[
        LazyAgent("anthropic", name="researcher", session=sess_a),
        LazyAgent("openai",    name="summariser", session=sess_a),
    ],
)

# Inner pipeline B: fact-checking (single agent, exposed as tool)
sess_b = LazySession()
fact_tool = LazyAgent("anthropic", name="checker", session=sess_b).as_tool(
    name="fact_check",
    description="Verify claims and return a fact-check report.",
)

# Outer orchestrator
master = LazyAgent(
    "anthropic",
    system="You coordinate research and fact-checking for comprehensive reports.",
)
result = master.loop(
    "Produce a verified report on quantum computing breakthroughs in 2024.",
    tools=[research_tool, fact_tool],
    max_steps=10,
)
print(result.content)
```
