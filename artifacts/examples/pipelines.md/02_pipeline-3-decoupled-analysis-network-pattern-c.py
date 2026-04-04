# Source   : lazy_wiki/human/pipelines.md
# Heading  : Pipeline 3 — Decoupled Analysis (Network, Pattern C)
# ID       : lazy_wiki/human/pipelines.md::pipeline-3-decoupled-analysis-network-pattern-c::00
# Kind     : context
# Testable : smoke_exec

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
