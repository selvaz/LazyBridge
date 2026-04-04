# Source   : lazy_wiki/human/context.md
# Heading  : Composing contexts
# ID       : lazy_wiki/human/context.md::composing-contexts::01
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyContext

ctx = LazyContext.merge(
    LazyContext.from_text("You are a senior analyst."),
    LazyContext.from_agent(data_collector),
    LazyContext.from_store(sess.store, keys=["market_data"]),
)
