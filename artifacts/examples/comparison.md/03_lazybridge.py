# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::01
# Kind     : llm_loop
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

result = LazyAgent("openai").loop(
    "What's the weather in Rome and Paris?",
    tools=[LazyTool.from_function(get_weather)],
)
print(result.content)
