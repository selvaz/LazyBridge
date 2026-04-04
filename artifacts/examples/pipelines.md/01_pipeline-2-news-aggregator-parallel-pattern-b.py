# Source   : lazy_wiki/human/pipelines.md
# Heading  : Pipeline 2 — News Aggregator (Parallel, Pattern B)
# ID       : lazy_wiki/human/pipelines.md::pipeline-2-news-aggregator-parallel-pattern-b::00
# Kind     : session_parallel
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession

sess = LazySession()

# Parallel tool: all three agents receive the same task and results are concatenated
gather_news = sess.as_tool(
    "gather_global_news",
    "Simultaneously gather AI news from the US, Europe, and Asia",
    mode="parallel",
    participants=[
        LazyAgent("anthropic", name="us_news",   session=sess),
        LazyAgent("openai",    name="eu_news",   session=sess),
        LazyAgent("google",    name="asia_news", session=sess),
    ],
    combiner="concat",
)

editor = LazyAgent("anthropic", name="editor", session=sess)
newsletter = editor.loop(
    "Gather today's AI news from the US, Europe, and Asia, then write a 400-word global digest.",
    tools=[gather_news],
)
print(newsletter.content)
