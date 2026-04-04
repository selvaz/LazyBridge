# Source   : lazy_wiki/human/pipelines.md
# Heading  : Pipeline 5 — Nested Pipelines (Pipeline as Tool)
# ID       : lazy_wiki/human/pipelines.md::pipeline-5-nested-pipelines-pipeline-as-tool::00
# Kind     : session_chain
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession

# Inner pipeline A: research → summarise (chain: summariser receives researcher's output)
sess_a = LazySession()
research_tool = sess_a.as_tool(
    "research",
    "Deep-research any topic and return a concise summary.",
    mode="chain",
    participants=[
        LazyAgent("anthropic", name="researcher", session=sess_a),
        LazyAgent("openai",    name="summariser", session=sess_a),
    ],
)

# Inner pipeline B: fact-checking (single agent, exposed as tool)
sess_b = LazySession()
fact_tool = LazyAgent("anthropic", name="checker", session=sess_b).as_tool(
    name="fact_check",
    description="Verify claims and return a fact-check report.",
)

# Outer orchestrator
master = LazyAgent(
    "anthropic",
    system="You coordinate research and fact-checking for comprehensive reports.",
)
result = master.loop(
    "Produce a verified report on quantum computing breakthroughs in 2024.",
    tools=[research_tool, fact_tool],
    max_steps=10,
)
print(result.content)
