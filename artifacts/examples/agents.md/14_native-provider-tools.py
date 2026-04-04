# Source   : lazy_wiki/human/agents.md
# Heading  : Native provider tools
# ID       : lazy_wiki/human/agents.md::native-provider-tools::00
# Kind     : native_tools
# Testable : full_exec

from lazybridge.core.types import NativeTool

resp = ai.chat(
    "What happened in AI this week?",
    native_tools=[NativeTool.WEB_SEARCH],
)
print(resp.content)

# Citations are available in grounding_sources
for src in resp.grounding_sources:
    print(src.url, src.title)
