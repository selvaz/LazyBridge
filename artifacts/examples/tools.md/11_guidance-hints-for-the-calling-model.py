# Source   : lazy_wiki/human/tools.md
# Heading  : Guidance — hints for the calling model
# ID       : lazy_wiki/human/tools.md::guidance-hints-for-the-calling-model::00
# Kind     : local
# Testable : full_exec

from lazybridge import LazyTool

tool = LazyTool.from_function(
    get_weather,
    guidance="Call this before answering any weather question. Always ask for the city if not provided.",
)
