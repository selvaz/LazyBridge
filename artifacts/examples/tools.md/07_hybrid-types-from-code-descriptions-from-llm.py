# Source   : lazy_wiki/human/tools.md
# Heading  : HYBRID — types from code, descriptions from LLM
# ID       : lazy_wiki/human/tools.md::hybrid-types-from-code-descriptions-from-llm::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyTool

from lazybridge import LazyAgent

llm  = LazyAgent("anthropic")
tool = LazyTool.from_function(
    search_web,
    schema_mode=ToolSchemaMode.HYBRID,
    schema_llm=llm,
)
