# Source   : lazy_wiki/human/tools.md
# Heading  : Structured output from a delegated agent
# ID       : lazy_wiki/human/tools.md::structured-output-from-a-delegated-agent::00
# Kind     : structured_output
# Testable : smoke_exec

from lazybridge import LazyTool

from pydantic import BaseModel

class ResearchResult(BaseModel):
    topic: str
    summary: str
    sources: list[str]

research_tool = LazyTool.from_agent(
    researcher,
    output_schema=ResearchResult,
)
