# Source   : lazy_wiki/human/sessions.md
# Heading  : Persistent sessions
# ID       : lazy_wiki/human/sessions.md::persistent-sessions::00
# Kind     : context
# Testable : smoke_exec

# Run 1
sess = LazySession(db="project.db")
researcher = LazyAgent("anthropic", session=sess)
researcher.loop("research phase 1")
sess.store.write("phase1", researcher.result)
# process exits — data saved to project.db

# Run 2 — pick up where you left off
from lazybridge import LazyAgent, LazySession, LazyContext  # agents must be reconstructed each run

sess2 = LazySession(db="project.db")
writer = LazyAgent("openai", session=sess2)
ctx = LazyContext.from_store(sess2.store, keys=["phase1"])
writer.chat("continue from this research", context=ctx)
