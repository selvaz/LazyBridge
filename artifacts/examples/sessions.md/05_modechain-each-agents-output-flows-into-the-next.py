# Source   : lazy_wiki/human/sessions.md
# Heading  : mode="chain" — each agent's output flows into the next
# ID       : lazy_wiki/human/sessions.md::modechain-each-agents-output-flows-into-the-next::00
# Kind     : session_chain
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession

sess = LazySession()
pipeline_tool = sess.as_tool(
    "research_pipeline",
    "Researches a topic, then produces an analysis. Returns the analyst's report.",
    mode="chain",
    participants=[
        LazyAgent("anthropic", name="researcher", session=sess),
        LazyAgent("openai",    name="analyst",    session=sess),
    ],
)

master = LazyAgent("anthropic")
master.loop("Analyse these 3 topics: fusion energy, quantum computing, biotech", tools=[pipeline_tool])
