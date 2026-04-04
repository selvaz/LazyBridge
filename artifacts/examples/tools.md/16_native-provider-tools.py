# Source   : lazy_wiki/human/tools.md
# Heading  : Native provider tools
# ID       : lazy_wiki/human/tools.md::native-provider-tools::00
# Kind     : native_tools
# Testable : full_exec

from lazybridge import LazyAgent
from lazybridge.core.types import NativeTool

ai = LazyAgent("anthropic")
resp = ai.chat(
    "What are the top AI research papers published this week?",
    native_tools=[NativeTool.WEB_SEARCH],
)
print(resp.content)

# You can also pass the string directly
resp = ai.chat("Latest news on fusion energy", native_tools=["web_search"])
