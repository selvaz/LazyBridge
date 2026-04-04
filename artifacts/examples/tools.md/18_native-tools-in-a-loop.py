# Source   : lazy_wiki/human/tools.md
# Heading  : Native tools in a loop
# ID       : lazy_wiki/human/tools.md::native-tools-in-a-loop::00
# Kind     : native_tools
# Testable : full_exec

from lazybridge.core.types import NativeTool

result = ai.loop(
    "Research the pros and cons of three different EV battery technologies",
    native_tools=[NativeTool.WEB_SEARCH],
    max_steps=10,
)
print(result.content)
