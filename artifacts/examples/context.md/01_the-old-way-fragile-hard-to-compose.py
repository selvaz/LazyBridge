# Source   : lazy_wiki/human/context.md
# Heading  : The old way — fragile, hard to compose
# ID       : lazy_wiki/human/context.md::the-old-way-fragile-hard-to-compose::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyAgent

from lazybridge import LazyContext

ctx = (
    LazyContext.from_text("You are a professional writer.")
    + LazyContext.from_agent(researcher)        # evaluates researcher's output when invoked
    + LazyContext.from_function(get_style_guide) # calls get_style_guide() when invoked
)

writer = LazyAgent("openai", context=ctx)
writer.chat("write an article")   # ctx is evaluated here, after researcher has run
