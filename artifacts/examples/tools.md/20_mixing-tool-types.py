# Source   : lazy_wiki/human/tools.md
# Heading  : Mixing tool types
# ID       : lazy_wiki/human/tools.md::mixing-tool-types::00
# Kind     : llm_loop
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyTool
from lazybridge.core.types import ToolDefinition

# A normal LazyTool with a Python callable
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

my_lazy_tool = LazyTool.from_function(search_web)

# A raw ToolDefinition for a tool handled externally (no Python callable)
raw = ToolDefinition(
    name="legacy_api",
    description="Query our legacy internal API",
    parameters={"type": "object", "properties": {"endpoint": {"type": "string"}}, "required": ["endpoint"]},
)

def call_legacy_api(endpoint: str) -> str:
    return f"Response from {endpoint}"

def tool_runner(name: str, args: dict):
    if name == "legacy_api":
        return call_legacy_api(args["endpoint"])
    raise RuntimeError(f"Unknown tool: {name}")

ai = LazyAgent("anthropic")
result = ai.loop(
    "Search for AI news and then query the /ai-index endpoint",
    tools=[my_lazy_tool, raw],
    tool_runner=tool_runner,   # called for tools without a LazyTool callable
)
