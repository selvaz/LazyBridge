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

**Session-free alternative:** if you don't need a session, `LazyTool.parallel()` is equivalent:

```python
from lazybridge import LazyAgent, LazyTool

gather_news = LazyTool.parallel(
    LazyAgent("anthropic", name="us_news"),
    LazyAgent("openai",    name="eu_news"),
    LazyAgent("google",    name="asia_news"),
    name="gather_global_news",
    description="Simultaneously gather AI news from US, Europe, and Asia",
    combiner="concat",
    concurrency_limit=3,   # optional: max simultaneous API calls (useful against rate limits)
    step_timeout=30.0,     # optional: per-agent timeout in seconds
)
# same tool, no session required
```

> **Cloning:** agents are cloned per invocation — `us_news._last_output` is `None` after the run. Use the editor's response or the tool's return value.

---

## Pipeline 3 — Decoupled Analysis (Network, Pattern C)

Three agents that don't know about each other. Communication flows through `LazyStore`.

```python
from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore(db="analysis.db")  # persistent

# --- Phase 1: Data Collection ---
collector = LazyAgent("anthropic", name="collector")
collector.loop("List the top 10 Python packages by GitHub stars in 2024.")
store.write("raw_data", collector.result, agent_id=collector.id)
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

## Pipeline 4a — Self-checking Loop (verify=)

The simplest review pattern: `loop()` retries automatically when the output doesn't pass a quality check. No reviewer agent, no router, no loop management.

```python
from lazybridge import LazyAgent

drafter = LazyAgent(
    "anthropic",
    system="You are a precise technical writer. Be accurate and concise.",
)

judge = LazyAgent(
    "anthropic",
    system=(
        "You are a quality reviewer. "
        "Reply 'approved' if the summary is accurate, self-contained, "
        "and exactly 200 words. Otherwise reply 'rejected: <reason>'."
    ),
)

result = drafter.loop(
    "Write a 200-word intro to transformer architecture.",
    verify=judge,
    max_verify=3,   # retry up to 3 times before accepting as-is
)
print(result.content)
```

The judge agent sees each draft and replies `approved` or `rejected: <reason>`. On rejection, `loop()` re-runs with the judge's reason appended as feedback. On approval (or after `max_verify` attempts), returns the current output.

> **Exhaustion warning:** if all `max_verify` attempts are rejected, `loop()` emits a `UserWarning` — `"loop() verify exhausted after N attempt(s) without approval."` — and returns the last result unchanged. No exception is raised.

*Use this when*: the review is a quality gate on a single agent's output — accuracy, length, format, policy compliance.

---

## Pipeline 4b — Multi-destination Router

Use `LazyRouter` when the review determines *which downstream agent runs next*, not just pass/fail. Here a reviewer routes to either a publisher (approved) or back to the drafter (needs revision).

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

draft = drafter.chat("Write a 200-word intro to transformer architecture.")

for revision in range(4):
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

*Use this when*: different review outcomes send the task to different agents — not just retry vs. accept.

---

## Pipeline 5 — Nested Pipelines (Pipeline as Tool)

An outer orchestrator manages two inner pipelines, each exposed as a single tool. Inner pipelines are declared with `as_tool(mode="chain")` — no wrapper functions, no manual context wiring.

```python
from lazybridge import LazyAgent, LazySession

# Inner pipeline A: research → summarise
# Option 1: with session (tracks events, uses existing session agents)
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

# Option 2: session-free (simpler, no tracking overhead)
# research_tool = LazyTool.chain(
#     LazyAgent("anthropic", name="researcher"),
#     LazyAgent("openai",    name="summariser"),
#     name="research",
#     description="Deep-research any topic and return a concise summary.",
#     step_timeout=60.0,   # optional: per-step timeout in seconds
# )
# chain() is async-under-the-hood — uses achat/aloop, never blocks the event loop

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

---

## Pipeline 6 — Local Documentation Skill

Index a folder of local documentation once, then query it from any agent using
BM25 retrieval. No vector database, no embeddings API — everything runs locally.

See [`lazy_wiki/human/tools.md`](tools.md) for the full guide.

```python
from lazybridge.ext.doc_skills import build_skill, skill_tool, skill_pipeline
from lazybridge import LazyAgent

# Step 1 — build the skill bundle (run once, or when docs change)
meta = build_skill(
    source_dirs=["./docs", "./reference"],
    skill_name="my-project",
    description="API reference and guides for MyProject.",
)

# Step 2a — single-step: agent calls the skill tool directly
tool  = skill_tool(meta["skill_dir"])
agent = LazyAgent("anthropic")
resp  = agent.loop("How do I configure retry behaviour?", tools=[tool])
print(resp.content)

# Step 2b — two-step pipeline: router sharpens the query, executor synthesises
pipeline     = skill_pipeline(skill_dir=meta["skill_dir"], provider="anthropic")
orchestrator = LazyAgent("anthropic")
resp = orchestrator.loop(
    "What is the canonical pattern for a sequential pipeline?",
    tools=[pipeline],
)
print(resp.content)
```

The pipeline is wired as `sess.as_tool(mode="chain")`:

```
user task
    │
    ▼
skill_router   — rewrites query for optimal BM25 retrieval; preserves technical names
    │
    ▼
skill_executor — calls skill tool (BM25, local) → synthesises grounded answer
    │
    ▼
orchestrator
```

See [`lazy_wiki/human/tools.md`](tools.md) for the full API reference.

---

## Pipeline 7 — Checkpoint & Resume (crash-safe pipelines)

Long pipelines can crash mid-run. Attach a `LazyStore` to `LazyTool.chain()` and set a `chain_id` — completed steps are saved automatically and skipped on the next run.

```python
from lazybridge import LazyAgent, LazyTool, LazyStore

store = LazyStore(db="pipeline.db")  # SQLite — survives process restarts

researcher = LazyAgent("anthropic", name="researcher")
analyst    = LazyAgent("openai",    name="analyst")
writer     = LazyAgent("anthropic", name="writer")

pipeline = LazyTool.chain(
    researcher, analyst, writer,
    name="report",
    description="Research → analyse → write",
    store=store,
    chain_id="report-v1",   # change this to invalidate old checkpoints
)

# Run 1: crashes after analyst finishes (e.g. API outage, process kill)
pipeline.run({"task": "Analyse the EV battery market in 2025"})
# ... process dies at writer step ...

# Run 2: researcher and analyst are skipped automatically
pipeline.run({"task": "Analyse the EV battery market in 2025"})
# Only writer runs — picks up from where it left off
```

**Concurrent runs:** if the same pipeline runs in parallel workers, give each worker its own checkpoint lane via `run_id=`:

```python
# Worker A — default lane
pipeline_a = LazyTool.chain(
    researcher, writer,
    name="pipe", description="d",
    store=store, chain_id="pipe-v1",
)

# Worker B — isolated lane (does not collide with Worker A)
pipeline_b = LazyTool.chain(
    researcher, writer,
    name="pipe", description="d",
    store=store, chain_id="pipe-v1", run_id="worker-b",
)
```

The same `run_id=` parameter is available on `sess.as_tool(mode="chain", run_id=...)`.

---

## Pipeline 8 — Saving & Loading Pipelines

### Save a tool to a file

`LazyTool.from_function()` and `agent.as_tool()` tools can be saved to human-readable Python files and loaded in a different process or on a different machine.

```python
from lazybridge import LazyAgent, LazyTool

def search_web(query: str) -> str:
    """Search the web for current information."""
    import httpx
    return httpx.get(f"https://api.example.com/search?q={query}").text

# Function-backed tool
tool = LazyTool.from_function(search_web)
tool.save("tools/search_web.py")

# Load in another script / worker process
tool2 = LazyTool.load("tools/search_web.py")
result = tool2.run({"query": "EV battery technology 2025"})
```

The saved file includes the function source, necessary imports, and the `LazyTool.from_function()` call. A sentinel comment `# LAZYBRIDGE_GENERATED_TOOL v1` at the top is required for `load()` to accept it.

```python
# Agent-backed tool — saves the agent constructor + as_tool() call
# API keys are NOT serialised
researcher = LazyAgent("anthropic", name="researcher",
                       system="You are a research analyst.")
researcher.as_tool("research", "Find factual information").save("tools/researcher.py")

loaded = LazyTool.load("tools/researcher.py")
```

**Limitations:**
- `LazyTool.chain()` and `LazyTool.parallel()` cannot be saved directly (`save()` raises `ValueError`). Save the individual participant tools instead.
- `LazyTool.load` must never be exposed as an agent tool — it executes arbitrary Python files.

### Save session topology (GraphSchema)

A `LazySession`'s graph topology (which agents exist, how they connect) can be serialised independently of the live Python objects:

```python
from lazybridge import LazyAgent, LazySession
from lazybridge import GraphSchema

sess = LazySession(db="project.db")
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

# Save topology
sess.graph.save("pipeline.json")     # auto-detects .json
sess.graph.save("pipeline.yaml")     # or .yaml (requires PyYAML)

# Reload as a descriptor (not live agents)
graph = GraphSchema.from_file("pipeline.json")
```

### Resume a session from SQLite

If you used `LazySession(db="pipeline.db")`, the session id and event log are persisted. Resume in a new process:

```python
from lazybridge import LazySession

# Resume the most recent session from the database
sess = LazySession.from_db("pipeline.db")

# Resume a specific session by id
sess = LazySession.from_db("pipeline.db", session_id="abc-123")

# Reconstruct graph topology from a JSON snapshot
sess = LazySession.from_json(json_str, db="pipeline.db")
```

---

## Pipeline 9 — GUI Tracking & Live Inspection

`lazybridge.gui` opens a browser panel for any LazyBridge object. No extra dependencies — pure stdlib.

```python
import lazybridge.gui   # activates .gui() on all core classes
from lazybridge import LazyAgent, LazySession, LazyTool

sess = LazySession(db="tracked.db", tracking="verbose")
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

pipeline = LazyTool.chain(researcher, writer,
                          name="article", description="Research then write")

# Open the GUI — one call registers all agents + the pipeline
sess.gui()       # SessionPanel + auto-registers AgentPanels
pipeline.gui()   # PipelinePanel with live per-step timeline

# Run — watch the browser tab update in real time
result = pipeline.run({"task": "Summarise fusion energy breakthroughs in 2025"})
print(result)
```

**What each panel shows:**

| Panel | What you can do |
|---|---|
| **AgentPanel** | Live-edit system prompt, model, tools. Run test calls with token + cost display. |
| **PipelinePanel** | See chain/parallel topology. Run with per-step timeline (start/finish events). |
| **SessionPanel** | Browse registered agents + store keys. Click any agent to open its panel. |
| **StorePanel** | Read, write, delete store keys. See which agent wrote each key + timestamp. |
| **RouterPanel** | Test routing logic — enter a value, see which agent is selected. |

**Live editing** applies to the running Python objects — edits to a system prompt take effect on the agent's next call, without restarting the process.

**Edits are not persisted to disk.** Use the "Export as Python" button in the AgentPanel to capture the current state as a code snippet.

For complete GUI reference: [`lazy_wiki/bot/16_gui.md`](../bot/16_gui.md).
