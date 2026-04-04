# Source   : lazy_wiki/human/tools.md
# Heading  : Ready to use immediately — no LLM schema call needed
# ID       : lazy_wiki/human/tools.md::ready-to-use-immediately-no-llm-schema-call-needed::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool
import json

tool = LazyTool.load("search_tool.json", fn=search_web)
result = tool.run({"query": "fusion energy"})
