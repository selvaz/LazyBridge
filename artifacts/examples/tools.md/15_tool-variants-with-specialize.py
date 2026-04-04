# Source   : lazy_wiki/human/tools.md
# Heading  : Tool variants with specialize()
# ID       : lazy_wiki/human/tools.md::tool-variants-with-specialize::00
# Kind     : llm_loop
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyTool

base = LazyTool.from_function(search_web, description="Search the web")

eu_search = base.specialize(
    name="search_eu",
    description="Search European news sources",
    guidance="Always filter to European sources published in the last 48 hours.",
)

us_search = base.specialize(
    name="search_us",
    description="Search US news sources",
    guidance="Always filter to US sources published in the last 48 hours.",
)

orchestrator = LazyAgent("anthropic")
orchestrator.loop(
    "Compare today's AI news coverage in the EU vs US",
    tools=[eu_search, us_search],
)
