# Source   : lazy_wiki/human/agents.md
# Heading  : as_tool() — expose this agent to another
# ID       : lazy_wiki/human/agents.md::as-tool-expose-this-agent-to-another::00
# Kind     : delegation
# Testable : smoke_exec

from lazybridge import LazyAgent

researcher = LazyAgent("anthropic", name="researcher", description="Researches any topic online")
research_tool = researcher.as_tool()

# Optional overrides
research_tool = researcher.as_tool(
    name="web_researcher",
    description="Deep-dives into a topic using multiple web searches",
    guidance="Use this whenever fresh external information is needed.",
)

orchestrator = LazyAgent("anthropic")
result = orchestrator.loop("Write a report on climate tech startups", tools=[research_tool])
