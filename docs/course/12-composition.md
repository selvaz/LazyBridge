# Agent Composition Patterns

This guide covers the core design philosophy of LazyBridge: agents as composable building blocks. Understand when to nest agents, when to chain them, when to let the LLM decide, and when to wire things manually.

---

## The spectrum: from manual to fully autonomous

```
Manual code          Chain pipeline         Orchestrator + tools
(you control)        (fixed order)          (LLM controls)
─────────────────────────────────────────────────────────────►
More predictable                              More flexible
```

---

## Pattern 1: Manual wiring (you are the orchestrator)

Write sequential Python code. You decide order, pass context explicitly.

```python
from lazybridge import LazyAgent, LazyContext

researcher = LazyAgent("anthropic", name="researcher", tools=[search])
writer = LazyAgent("openai", name="writer")

# You decide: researcher first, then writer
resp1 = researcher.loop("Find AI safety papers from 2025")
resp2 = writer.chat(
    "Write a blog post about these findings",
    context=LazyContext.from_agent(researcher),
)
print(resp2.content)
```

**When to use:** Simple 2-3 step flows where the order never changes and you want full control. Good for scripts and notebooks.

**Tradeoff:** No checkpoints, no observability, not reusable as a tool.

---

## Pattern 2: Chain pipeline (fixed order, framework-managed)

Same flow, but the framework handles context passing, checkpoints, and error recovery.

```python
from lazybridge import LazyAgent, LazyTool, LazyStore

researcher = LazyAgent("anthropic", name="researcher", tools=[search])
writer = LazyAgent("openai", name="writer")

pipeline = LazyTool.chain(
    researcher, writer,
    name="blog_pipeline",
    description="Research then write",
    store=LazyStore(db="blog.db"),  # checkpoint persistence
    chain_id="blog-v1",
)

result = pipeline.run({"task": "AI safety blog post"})
```

**When to use:** Fixed multi-step workflows in production. You want checkpoints, resume-on-crash, and the ability to expose the pipeline as a tool.

**Tradeoff:** Fixed order — can't skip steps or go back.

---

## Pattern 3: Orchestrator with agent-tools (LLM decides)

An orchestrator agent receives specialized agents as tools. The LLM decides which to call, in what order, and how many times.

```python
researcher = LazyAgent("anthropic", name="researcher", tools=[search])
analyst = LazyAgent("openai", name="analyst", tools=[calculator])
writer = LazyAgent("anthropic", name="writer")

orchestrator = LazyAgent("anthropic",
    system="You coordinate research projects. Use tools strategically.",
)
resp = orchestrator.loop(
    "Research AI safety, analyze the key risks, then write a report",
    tools=[
        researcher.as_tool("research", "Find information on a topic"),
        analyst.as_tool("analyze", "Run calculations and analysis"),
        writer.as_tool("write", "Write polished content"),
    ],
)
```

**When to use:** Complex tasks where the optimal order depends on the input. The LLM can call researcher twice if the first search wasn't good enough, skip the analyst if no numbers are needed, etc.

**Tradeoff:** Less predictable cost and behavior. The LLM may over-call tools.

---

## Pattern 4: Forced tool use with `tool_choice`

Make agents that MUST use their tools before responding:

```python
# This researcher always searches — never answers from memory
researcher = LazyAgent("anthropic", tools=[web_search, arxiv])
research_tool = researcher.as_tool("research", "Search for info",
                                    tool_choice="required")

# This analyst always runs calculations
analyst = LazyAgent("openai", tools=[calculator, stats])
analysis_tool = analyst.as_tool("analyze", "Compute metrics",
                                 tool_choice="required")

# Orchestrator uses them — each agent is forced to use its tools
orchestrator = LazyAgent("anthropic")
orchestrator.loop("Analyze revenue data", tools=[research_tool, analysis_tool])
```

**When to use:** When an agent's value comes from its tools, not its general knowledge. Researchers should always search. Calculators should always compute.

### Parallel tool execution

When tools are I/O-bound (API calls, web requests), run them concurrently:

```python
# Orchestrator fires 3 lookups at once
resp = await orchestrator.aloop(
    "Get weather for NYC, London, and Tokyo",
    tools=[weather_tool],
    tool_choice="parallel",  # all tool calls in a step run via asyncio.gather()
)
```

---

## Pattern 5: Nested pipelines as tools

The most powerful pattern — pipelines inside pipelines:

```python
# Level 2: specialized pipelines
research_pipe = LazyTool.chain(
    data_fetcher, analyst,
    name="research", description="Fetch data then analyze",
)

content_pipe = LazyTool.chain(
    writer, editor, translator,
    name="content", description="Write, edit, translate",
)

# Level 1: orchestrator sees pipelines as atomic tools
orchestrator = LazyAgent("anthropic")
resp = orchestrator.loop(
    "Research Tesla financials, then produce a report in English and Italian",
    tools=[research_pipe, content_pipe],
)
```

What happens at runtime:
```
orchestrator (LLM decides order)
  ├── calls research_pipe(task="Tesla financials")
  │     ├── data_fetcher runs (with its own tools)
  │     └── analyst runs (with calculator, chart_tool)
  └── calls content_pipe(task="produce report in EN and IT")
        ├── writer runs
        ├── editor runs
        └── translator runs
```

Each level is autonomous. The orchestrator doesn't know the internal structure.

---

## Pattern 6: Session-based composition

Use a session when agents need shared state (store, events, graph):

```python
from lazybridge import LazySession

sess = LazySession(db="project.db", tracking="verbose")

# Agents share the session's store and event log
researcher = LazyAgent("anthropic", name="researcher",
                        tools=[search], session=sess)
writer = LazyAgent("openai", name="writer", session=sess)

# Option A: chain via session
pipeline = sess.as_tool("pipeline", "Research then write", mode="chain")
result = pipeline.run({"task": "AI trends report"})

# Option B: parallel via session
panel = sess.as_tool("panel", "Multi-perspective analysis", mode="parallel")
result = panel.run({"task": "Analyze AI regulation"})

# After running, inspect costs and events
print(sess.usage_summary())
print(sess.events.get(event_type="tool_call"))
```

**When to use:** Production pipelines where you need observability, cost tracking, shared data, or persistent graph topology.

---

## Pattern 7: Mixed composition

Real systems combine patterns. Here's a realistic example:

```python
from lazybridge import (
    LazyAgent, LazyTool, LazySession, LazyStore,
    ContentGuard, GuardAction, GuardChain,
    OTelExporter,
)

# ── Guards ──
pii_guard = ContentGuard(
    input_fn=lambda t: GuardAction.block("PII") if "@" in t else GuardAction.allow(),
    output_fn=lambda t: GuardAction.allow(),
)

# ── Session with observability ──
sess = LazySession(
    db="production.db",
    tracking="verbose",
    exporters=[OTelExporter(service_name="my-app")],
)

# ── Specialized agents (tools bound at construction) ──
researcher = LazyAgent("anthropic", name="researcher",
                        tools=[web_search, arxiv_search], session=sess)

analyst = LazyAgent("openai", name="analyst",
                     tools=[calculator, chart_tool], session=sess)

writer = LazyAgent("anthropic", name="writer", session=sess)

# ── Build pipelines ──
research_pipe = LazyTool.chain(
    researcher, analyst,
    name="deep_research",
    description="Search then analyze",
    store=sess.store,
    chain_id="research",
)

# ── Judge for quality gate ──
judge = LazyAgent("openai", model="gpt-4o-mini")

# ── Orchestrator: combines everything ──
orchestrator = LazyAgent("anthropic", name="orchestrator",
    system="You coordinate research projects. Be thorough.",
    session=sess,
)

result = orchestrator.loop(
    "Produce a comprehensive AI safety report for Q1 2026",
    tools=[
        research_pipe,                                          # nested pipeline
        writer.as_tool("write", "Write content"),               # standalone agent
        researcher.as_tool("quick_search", "Quick web lookup",  # forced tool use
                           tool_choice="required"),
    ],
    verify=judge,       # quality gate
    max_verify=2,
    guard=pii_guard,    # input/output safety
)

print(f"Cost: ${sess.usage_summary()['total']['cost_usd']:.4f}")
```

---

## Decision matrix

| I need... | Use |
|-----------|-----|
| Simple A → B → C, always same order | `LazyTool.chain()` |
| Agents working on same task simultaneously | `LazyTool.parallel()` or `sess.as_tool(mode="parallel")` |
| LLM to decide which agents to call | Orchestrator + `as_tool()` |
| Agent to always use its tools | `as_tool(tool_choice="required")` |
| Multiple tool calls to run concurrently | `tool_choice="parallel"` |
| Pipeline as a building block for bigger systems | `LazyTool.chain()` → pass as tool to orchestrator |
| Shared state between agents | `LazySession` |
| Quality gate on final output | `verify=judge_agent` |
| Safety checks on input/output | `guard=my_guard` |
| Cost tracking and observability | `LazySession(tracking="verbose", exporters=[...])` |
| Quick 2-step script | Manual wiring with `LazyContext.from_agent()` |

---

## Anti-patterns

**Don't force chains when the LLM should decide:**
```python
# Bad — if analyst is sometimes unnecessary, this wastes tokens
pipeline = LazyTool.chain(researcher, analyst, writer)

# Better — orchestrator skips analyst when not needed
orchestrator.loop("task", tools=[research, analyze, write])
```

**Don't use orchestrator when order is always fixed:**
```python
# Bad — LLM might call writer before researcher
orchestrator.loop("task", tools=[research, write])

# Better — chain guarantees order
LazyTool.chain(researcher, writer)
```

**Don't pass tools at call level for specialized agents:**
```python
# Bad — tools leak into the orchestrator's concern
orchestrator.loop("task", tools=[search, calc, writer_tool])

# Better — each agent owns its tools
researcher = LazyAgent("anthropic", tools=[search])
analyst = LazyAgent("openai", tools=[calc])
orchestrator.loop("task", tools=[
    researcher.as_tool("r", "..."),
    analyst.as_tool("a", "..."),
])
```
