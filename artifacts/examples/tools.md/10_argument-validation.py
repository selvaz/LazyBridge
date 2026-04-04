# Source   : lazy_wiki/human/tools.md
# Heading  : Argument validation
# ID       : lazy_wiki/human/tools.md::argument-validation::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool

from lazybridge import ToolArgumentValidationError

def divide(a: int, b: int) -> float:
    return a / b

tool = LazyTool.from_function(divide)

# Type coercion: "5" → 5 automatically
tool.run({"a": "5", "b": "2"})   # works fine

# Validation error on truly invalid input
try:
    tool.run({"a": "not-a-number", "b": 2})
except ToolArgumentValidationError as e:
    print(e)
