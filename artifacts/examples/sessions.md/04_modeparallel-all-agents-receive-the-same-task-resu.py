# Source   : lazy_wiki/human/sessions.md
# Heading  : mode="parallel" — all agents receive the same task, results combined
# ID       : lazy_wiki/human/sessions.md::modeparallel-all-agents-receive-the-same-task-results-combined::00
# Kind     : session_parallel
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession

sess = LazySession()

gather_news = sess.as_tool(
    "gather_global_news",
    "Simultaneously gather AI news from the US, Europe, and Asia. Returns combined results.",
    mode="parallel",
    participants=[
        LazyAgent("anthropic", name="us_news",   session=sess),
        LazyAgent("openai",    name="eu_news",   session=sess),
        LazyAgent("google",    name="asia_news", session=sess),
    ],
    combiner="concat",   # join outputs with newlines (default)
)

editor = LazyAgent("anthropic", name="editor", session=sess)
editor.loop(
    "Gather today's AI news from the US, Europe, and Asia, then write a 400-word global digest.",
    tools=[gather_news],
)
