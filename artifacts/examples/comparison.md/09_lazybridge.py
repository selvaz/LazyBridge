# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::04
# Kind     : session_parallel
# Testable : smoke_exec

from lazybridge import LazyAgent, LazySession

sess    = LazySession()
regions = ["the United States", "Europe", "Asia"]

news_tool = sess.as_tool(
    "gather_news", "Simultaneously gather AI news from three regions",
    mode="parallel",
    participants=[
        LazyAgent("openai", system=f"Report AI news from {r} only.", session=sess)
        for r in regions
    ],
    combiner="concat",
)

digest = LazyAgent("openai", session=sess).loop(
    "Gather regional AI news summaries. Write a 200-word global digest.",
    tools=[news_tool],
)
print(digest.content)
