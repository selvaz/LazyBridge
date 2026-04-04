# Source   : lazy_wiki/human/quickstart.md
# Heading  : Example 3 — Multi-agent pipeline
# ID       : lazy_wiki/human/quickstart.md::example-3-multi-agent-pipeline::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession, LazyContext

# Shared container (tracking, store, graph)
sess = LazySession()

# Two agents connected to the same session
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

# Step 1: researcher does its work
researcher.loop("Find the top 3 developments in AI this week.")

# Step 2: writer reads researcher's output via LazyContext
writer.chat(
    "Write a short newsletter section from this research.",
    context=LazyContext.from_agent(researcher),
)

# Read the writer's result
print(writer.result)          # plain text output

# Inspect what happened
print(sess.events.get())      # full event log
print(sess.store.read_all())  # shared state
