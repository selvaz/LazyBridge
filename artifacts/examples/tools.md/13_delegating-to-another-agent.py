# Source   : lazy_wiki/human/tools.md
# Heading  : Delegating to another agent
# ID       : lazy_wiki/human/tools.md::delegating-to-another-agent::00
# Kind     : delegation
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyTool

researcher = LazyAgent(
    "anthropic",
    name="researcher",
    description="Researches any topic and returns a detailed summary.",
)

# Option A: from the agent
research_tool = researcher.as_tool()

# Option B: explicit factory
research_tool = LazyTool.from_agent(
    researcher,
    guidance="Use this for any question that requires current or detailed information.",
)

orchestrator = LazyAgent("anthropic")
result = orchestrator.loop(
    "Prepare a brief on open-source AI models released this year",
    tools=[research_tool],
)
