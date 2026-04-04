# Source   : lazy_wiki/human/agents.md
# Heading  : Conversazione con memoria
# ID       : lazy_wiki/human/agents.md::conversazione-con-memoria::01
# Kind     : local
# Testable : local_exec

print(len(mem))        # 4 — 2 user + 2 assistant messages
print(mem.history)     # list of {"role": ..., "content": ...} dicts
mem.clear()            # reset the conversation
