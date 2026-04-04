# Source   : lazy_wiki/human/quickstart.md
# Heading  : Example 2 — Tool loop (function calling)
# ID       : lazy_wiki/human/quickstart.md::example-2-tool-loop-function-calling::00
# Kind     : llm_loop
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # In a real app, call a weather API here
    return f"Weather in {city}: 22°C, partly cloudy"

weather_tool = LazyTool.from_function(get_weather)

ai = LazyAgent("anthropic")
result = ai.loop("What's the weather like in Rome and Paris?", tools=[weather_tool])
print(result.content)
