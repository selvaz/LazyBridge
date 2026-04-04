# Source   : lazy_wiki/human/context.md
# Heading  : Agent-level vs call-level context
# ID       : lazy_wiki/human/context.md::agent-level-vs-call-level-context::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyContext

# Context applied to every call on this agent:
writer = LazyAgent("openai", context=LazyContext.from_text("Write in Italian."))
writer.chat("What is Python?")    # "Python è un linguaggio..."

# Override for a single call:
writer.chat("What is Python?", context=LazyContext.from_text("Write in German."))
# call-level context replaces agent-level context for this call only
