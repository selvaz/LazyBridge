# Source   : lazy_wiki/human/context.md
# Heading  : Full example
# ID       : lazy_wiki/human/context.md::full-example::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyContext, LazyStore

store = LazyStore()

# Contexts declared once — evaluated lazily when each agent actually runs
collector = LazyAgent("anthropic", name="collector")
analyst   = LazyAgent("anthropic", name="analyst",
                      context=LazyContext.from_store(store, keys=["papers"]))
writer    = LazyAgent("openai",
                      context=(
                          LazyContext.from_text("Write for a non-technical audience.")
                          + LazyContext.from_store(store, keys=["findings"])
                      ))

# Pipeline: pure execution + store handoffs
collector.loop("Collect the top 5 AI papers this month")
store.write("papers", collector.result)

analyst.chat("Identify the 3 most impactful findings")
store.write("findings", analyst.result)

writer.chat("Write a blog post from these findings")
