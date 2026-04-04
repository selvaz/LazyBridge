# Source   : lazy_wiki/human/tools.md
# Heading  : Override name and description
# ID       : lazy_wiki/human/tools.md::override-name-and-description::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool

tool = LazyTool.from_function(
    get_weather,
    name="weather_checker",
    description="Look up the current weather in any city worldwide.",
)
