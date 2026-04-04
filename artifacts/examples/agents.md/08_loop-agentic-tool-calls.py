# Source   : lazy_wiki/human/agents.md
# Heading  : loop() — agentic tool calls
# ID       : lazy_wiki/human/agents.md::loop-agentic-tool-calls::00
# Kind     : llm_loop
# Testable : smoke_exec

from lazybridge import LazyTool

def search_web(query: str) -> str:
    """Search the web and return results."""
    return f"Results for '{query}': [article1, article2, ...]"

search = LazyTool.from_function(search_web)

result = ai.loop(
    "Find the latest news about fusion energy and summarise it.",
    tools=[search],
    max_steps=8,         # hard cap (default: 8)
)
print(result.content)
