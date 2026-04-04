# Source   : lazy_wiki/human/tools.md
# Heading  : LLM — full schema from the model
# ID       : lazy_wiki/human/tools.md::llm-full-schema-from-the-model::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool

tool = LazyTool.from_function(
    legacy_function,       # no type hints, no docstrings
    schema_mode=ToolSchemaMode.LLM,
    schema_llm=llm,
)
