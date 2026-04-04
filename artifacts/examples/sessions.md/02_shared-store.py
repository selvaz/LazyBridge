# Source   : lazy_wiki/human/sessions.md
# Heading  : Shared store
# ID       : lazy_wiki/human/sessions.md::shared-store::00
# Kind     : context
# Testable : smoke_exec

# Agent writes results
researcher.loop("find top AI news")
sess.store.write("ai_news", researcher.result, agent_id=researcher.id)

# Another agent reads without needing a reference to researcher
from lazybridge import LazyContext
ctx = LazyContext.from_store(sess.store, keys=["ai_news"])
writer.chat("write a newsletter section", context=ctx)

# Read everything
print(sess.store.read_all())

# Read only what a specific agent wrote
print(sess.store.read_by_agent(researcher.id))
