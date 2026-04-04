# Source   : lazy_wiki/human/tools.md
# Heading  : Using a session's pipeline as a tool
# ID       : lazy_wiki/human/tools.md::using-a-sessions-pipeline-as-a-tool::00
# Kind     : session_chain
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession

sess = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)
analyst    = LazyAgent("openai",    name="analyst",    session=sess)

# Chain: researcher runs first, analyst receives researcher's output
pipeline_tool = sess.as_tool(
    "research_and_analyse",
    "Researches a topic and produces an analysis. Returns the analyst's report.",
    mode="chain",
    participants=[researcher, analyst],
)

master = LazyAgent("anthropic")
master.loop("Investigate three topics for our quarterly report", tools=[pipeline_tool])
