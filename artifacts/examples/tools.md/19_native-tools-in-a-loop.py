# Source   : lazy_wiki/human/tools.md
# Heading  : Native tools in a loop
# ID       : lazy_wiki/human/tools.md::native-tools-in-a-loop::01
# Kind     : native_tools
# Testable : full_exec

from lazybridge.core.types import NativeTool

result = ai.loop(
    "Search for today's news on quantum computing, then format it as a structured report",
    tools=[format_report_tool],
    native_tools=[NativeTool.WEB_SEARCH],
)
