# Source   : lazy_wiki/human/tools.md
# Heading  : Pre-compiling the schema
# ID       : lazy_wiki/human/tools.md::pre-compiling-the-schema::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool

tool = LazyTool.from_function(
    search_web,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=llm,
).compile()   # LLM call happens here, not during the first agent loop
