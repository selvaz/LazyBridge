# Source   : lazy_wiki/human/tools.md
# Heading  : Wrapping a Python function
# ID       : lazy_wiki/human/tools.md::wrapping-a-python-function::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool

def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°{unit[0].upper()}, sunny"

tool = LazyTool.from_function(get_weather)
