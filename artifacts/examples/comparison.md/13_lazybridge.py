# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::06
# Kind     : native_tools
# Testable : full_exec

from lazybridge import LazyAgent
from lazybridge.core.types import NativeTool

resp = LazyAgent("anthropic").chat(
    "What happened in AI this week?",
    native_tools=[NativeTool.WEB_SEARCH],
)
print(resp.content)
for src in resp.grounding_sources:
    print(src.url, src.title)
