# Source   : lazy_wiki/human/tools.md
# Heading  : Using tools in a loop
# ID       : lazy_wiki/human/tools.md::using-tools-in-a-loop::00
# Kind     : llm_loop
# Testable : smoke_exec

from lazybridge import LazyAgent

ai = LazyAgent("anthropic")
result = ai.loop(
    "What's the weather in Rome, Paris, and Tokyo?",
    tools=[tool],
)
print(result.content)
