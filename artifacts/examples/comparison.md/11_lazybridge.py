# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::05
# Kind     : memory
# Testable : smoke_exec

from lazybridge import LazyAgent, Memory

ai  = LazyAgent("anthropic")
mem = Memory()

print(ai.text("My name is Marco",          memory=mem))
print(ai.text("What is my name?",          memory=mem))
print(ai.text("What did we discuss so far?", memory=mem))
