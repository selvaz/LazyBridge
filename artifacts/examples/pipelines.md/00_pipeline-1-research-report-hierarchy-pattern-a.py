# Source   : lazy_wiki/human/pipelines.md
# Heading  : Pipeline 1 — Research Report (Hierarchy, Pattern A)
# ID       : lazy_wiki/human/pipelines.md::pipeline-1-research-report-hierarchy-pattern-a::00
# Kind     : delegation
# Testable : smoke_exec

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
