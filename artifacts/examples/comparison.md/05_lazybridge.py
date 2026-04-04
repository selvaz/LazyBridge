# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::02
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyContext

researcher = LazyAgent("anthropic")
writer     = LazyAgent("openai", context=LazyContext.from_agent(researcher))

researcher.chat("Find AI news this week")
print(writer.text("Write a newsletter section"))
