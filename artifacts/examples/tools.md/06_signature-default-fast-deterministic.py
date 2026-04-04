# Source   : lazy_wiki/human/tools.md
# Heading  : SIGNATURE (default) — fast, deterministic
# ID       : lazy_wiki/human/tools.md::signature-default-fast-deterministic::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool, ToolSchemaMode

tool = LazyTool.from_function(search_web)                          # implicit
tool = LazyTool.from_function(search_web, schema_mode=ToolSchemaMode.SIGNATURE)  # explicit
