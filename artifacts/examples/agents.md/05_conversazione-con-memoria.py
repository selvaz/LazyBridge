# Source   : lazy_wiki/human/agents.md
# Heading  : Conversazione con memoria
# ID       : lazy_wiki/human/agents.md::conversazione-con-memoria::03
# Kind     : memory
# Testable : smoke_exec

import json
from lazybridge import LazyAgent, Memory, LazySession

sess = LazySession(db="chat.db")
ai  = LazyAgent("anthropic", session=sess)

# Restore previous session
raw = sess.store.read("history")
mem = Memory.from_history(json.loads(raw)) if raw else Memory()

ai.chat("continue from where we left off", memory=mem)

# Save at the end
sess.store.write("history", json.dumps(mem.history))
