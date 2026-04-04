# Source   : lazy_wiki/human/agents.md
# Heading  : Conversazione con memoria
# ID       : lazy_wiki/human/agents.md::conversazione-con-memoria::00
# Kind     : memory
# Testable : smoke_exec

from lazybridge import LazyAgent, Memory

ai  = LazyAgent("anthropic")
mem = Memory()

ai.chat("My name is Marco", memory=mem)
resp = ai.chat("What's my name?", memory=mem)
print(resp.content)   # "Marco"
