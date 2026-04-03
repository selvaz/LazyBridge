# Pipeline Patterns — Complete Runnable Code

## Choosing the right pattern

```
Do you need one agent to drive others, deciding what to call and when?
│
├─ YES → Pattern A (Hierarchy)
│         orchestrator.loop(..., tools=[agent_a.as_tool(), agent_b.as_tool()])
│
└─ NO
   │
   ├─ Do agents run independently on the same task, in parallel?
   │   │
   │   ├─ YES, and you want automatic wiring → Pattern B via sess.as_tool(mode="parallel")
   │   │                                        or manual: await sess.gather(a.achat(...), b.achat(...))
   │   │
   │   └─ YES, but decoupled (different loops, different scripts, persistent state)
   │             → Pattern C (Network via LazyStore)
   │
   ├─ Do agents run sequentially, each feeding output to the next?
   │   │
   │   └─ YES → Pattern B via sess.as_tool(mode="chain")
   │             or manual: result = a.chat(task); b.chat(result.content)
   │
   ├─ Do you need to branch to different agents based on a condition?
   │   │
   │   └─ YES → Pattern D (LazyRouter)
   │             router = LazyRouter(condition=fn, routes={...})
   │             next_agent = router.route(result)
   │
   └─ Do you need to expose a whole pipeline as a single tool to an outer orchestrator?
             → Pattern E (Pipeline as Tool)
               pipeline_tool = LazyTool.from_function(run_pipeline, ...)
```

### Which method to call on a single agent?

```
Do you need tool use (the agent calls Python functions)?
│
├─ YES → agent.loop() / agent.aloop()
│
└─ NO
   │
   ├─ Do you need structured output (Pydantic/JSON schema)?
   │   │
   │   ├─ YES → agent.json(msg, schema=MyModel) / agent.ajson(...)
   │   │         (native structured output + JSON enforcement in system prompt)
   │   │
   │   └─ NO
   │       │
   │       ├─ Do you need just the text string, not the full CompletionResponse?
   │       │   └─ YES → agent.text(msg) / agent.atext(...)
   │       │
   │       └─ NO → agent.chat(msg) / agent.achat(...)
   │                (returns CompletionResponse with .content, .usage, .tool_calls, etc.)
   │
   └─ Do you need streaming output?
               └─ agent.chat(msg, stream=True)  →  Iterator[StreamChunk]
                  agent.achat(msg, stream=True) →  AsyncIterator[StreamChunk]
```

---

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

Multiple agents run concurrently. Expose them as a single `mode="parallel"` tool — no asyncio boilerplate, no manual store writes, no context wiring.

**Communication**: the parallel tool concatenates outputs and returns them as a tool result.
**Use case**: fan-out research, parallel processing of different topics.

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession(tracking="verbose")

research_tool = sess.as_tool(
    "multi_domain_research",
    "Simultaneously research transformer architecture, market trends, and AI regulation",
    mode="parallel",
    participants=[
        LazyAgent("anthropic", name="tech_researcher",   session=sess),
        LazyAgent("openai",    name="market_researcher", session=sess),
        LazyAgent("anthropic", name="legal_researcher",  session=sess),
    ],
    combiner="concat",
)

editor = LazyAgent("anthropic", name="editor", session=sess)
report = editor.loop(
    "Research the current state of AI across technology, market, and regulation. Write an executive summary.",
    tools=[research_tool],
)
print(report.content)
```

**When you need the raw concurrent results** (e.g. to store them individually or inspect per-agent output), use `sess.gather()` directly:

```python
import asyncio
from lazybridge import LazyAgent, LazySession

sess = LazySession()
agent_tech   = LazyAgent("anthropic", name="tech",   session=sess)
agent_market = LazyAgent("openai",    name="market", session=sess)

async def main():
    tech, market = await sess.gather(
        agent_tech.aloop("Summarise transformer advances"),
        agent_market.aloop("Summarise AI market trends"),
    )
    sess.store.write("tech",   tech.content,   agent_id=agent_tech.id)
    sess.store.write("market", market.content, agent_id=agent_market.id)

asyncio.run(main())
```

---

## Pattern C — Network (Cross-loop, No Direct References)

Agents communicate via `LazyStore` (state) and `LazyContext.from_store()` (context injection). Agents don't need to know about each other. Declare contexts at construction — they are evaluated lazily when the agent actually runs.

**Use case**: long-running pipelines, agents in separate loops, persistent state across restarts.

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore(db="pipeline.db")   # persistent SQLite

# Declare contexts upfront — LazyContext.from_store reads the store at call time
collector = LazyAgent("anthropic", name="collector")
analyst   = LazyAgent("openai",    name="analyst",
                      context=LazyContext.from_store(store, keys=["papers"]))
writer    = LazyAgent("anthropic", name="writer",
                      context=LazyContext.from_store(store, keys=["papers", "analysis"]))

# Pipeline: each step writes to store; the next step's context picks it up automatically
collector.loop("Collect the top 5 AI papers published this week")
store.write("papers", collector._last_output, agent_id=collector.id)

analyst.chat("Identify the 3 most impactful findings from these papers.")
store.write("analysis", analyst._last_output, agent_id=analyst.id)

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

A full pipeline (LazySession) is exposed as a single tool to an external orchestrator. Use `as_tool(mode="chain")` — no wrapper functions, no manual context wiring.

```python
from lazybridge import LazyAgent, LazySession

# Inner pipeline: researcher feeds output directly to summariser
inner_sess = LazySession()
pipeline_tool = inner_sess.as_tool(
    "research_and_summarise",
    "Researches a topic and returns a concise summary.",
    mode="chain",
    participants=[
        LazyAgent("anthropic", name="researcher", session=inner_sess),
        LazyAgent("openai",    name="summariser", session=inner_sess),
    ],
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

---

## Pattern F — Lazy Context Chaining (from_agent)

Agents read each other's outputs through `LazyContext.from_agent()`. The entire topology — who sees what — is declared at construction time. No agent runs until you explicitly call it.

**Key properties:**
- **Pull, not push.** Agent A does not send anything. Agent B reads `A._last_output` when B is called.
- **Safe before execution.** If A hasn't run yet, context is `""` — no crash, just a debug log.
- **Composable.** Combine with `from_store()`, `from_text()`, `from_function()` using `+`.
- **Inspectable.** Call `agent.context()` to see exactly what the LLM will receive before running.

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore()
store.write("style_guide", "Formal tone. Max 300 words. Cite all sources.")

# ── Declare topology — zero side effects ──────────────────────────────────
researcher = LazyAgent("anthropic", name="researcher")

fact_checker = LazyAgent("openai", name="fact_checker",
    context=LazyContext.from_agent(researcher))

writer = LazyAgent("anthropic", name="writer",
    context=(
        LazyContext.from_agent(researcher,    prefix="[Research]")
        + LazyContext.from_agent(fact_checker, prefix="[Fact-check]")
        + LazyContext.from_store(store, keys=["style_guide"])
    ))

# ── Execute in order ──────────────────────────────────────────────────────
researcher.loop("Key findings in fusion energy research this year?")
fact_checker.chat("Verify the claims. Flag any missing citations.")
result = writer.chat("Write the briefing using your context.")
print(result.content)
```

**When to use this pattern:**
- Linear pipelines where each step enriches the next
- Any time you want "agent B sees what agent A said" without passing return values manually
- Declaring complex topologies in config or session constructors — agents are wired before any call

**When to prefer return values instead:**
- The downstream agent needs to *act on* the data (branch, transform, validate)
- You need strong typing — use `output_schema` + `from_agent()` together
- The pipeline is non-linear (fan-out/fan-in) — use `LazySession.as_tool(mode="parallel")` + store

**Combining with from_store for decoupled pipelines:**

```python
# Agent A writes to store; Agent B reads from store via context
# They never reference each other — fully decoupled
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore()

a = LazyAgent("anthropic", name="extractor", session_store=store)
b = LazyAgent("openai", name="summariser",
    context=LazyContext.from_store(store, keys=["extraction"]))

# A runs and writes to store
extraction = a.chat("Extract all named entities from the document.")
store.write("extraction", extraction.content)

# B reads from store — never knew about A
result = b.chat("Summarise the extracted entities.")
```
