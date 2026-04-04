# Source   : lazy_wiki/human/agents.md
# Heading  : Conversazione con memoria
# ID       : lazy_wiki/human/agents.md::conversazione-con-memoria::02
# Kind     : memory
# Testable : smoke_exec

from lazybridge import Memory

mem = Memory()
agent_a.chat("Remember: the project deadline is Friday", memory=mem)
agent_b.chat("What's the deadline?", memory=mem)   # answers "Friday"
