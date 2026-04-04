# Source   : lazy_wiki/human/sessions.md
# Heading  : Creating a session
# ID       : lazy_wiki/human/sessions.md::creating-a-session::00
# Kind     : persistence
# Testable : full_exec

from lazybridge import LazySession

# In-memory (default) — data lost when process exits
sess = LazySession()

# Persistent — SQLite file stores events and state
sess = LazySession(db="my_pipeline.db")

# Control tracking verbosity
sess = LazySession(tracking="off")      # no tracking
sess = LazySession(tracking="basic")    # default: all events except stream chunks
sess = LazySession(tracking="verbose")  # everything including stream chunks
