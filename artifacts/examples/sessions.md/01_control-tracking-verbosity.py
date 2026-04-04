# Source   : lazy_wiki/human/sessions.md
# Heading  : Control tracking verbosity
# ID       : lazy_wiki/human/sessions.md::control-tracking-verbosity::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)
# Both agents now share sess.store, sess.events, and appear in sess.graph
