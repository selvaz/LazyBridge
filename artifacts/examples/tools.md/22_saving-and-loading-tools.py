# Source   : lazy_wiki/human/tools.md
# Heading  : Saving and loading tools
# ID       : lazy_wiki/human/tools.md::saving-and-loading-tools::00
# Kind     : local
# Testable : full_exec

import json

# Save
tool = LazyTool.from_function(
    search_web,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=llm,
).compile()   # generate the schema now

tool.save("search_tool.json")

# Load in a different process or later run
from lazybridge import LazyTool

tool = LazyTool.load("search_tool.json")
# Ready to use immediately — no LLM schema call needed
result = tool.run({"query": "latest AI news"})
