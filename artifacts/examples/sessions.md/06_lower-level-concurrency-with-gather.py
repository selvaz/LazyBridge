# Source   : lazy_wiki/human/sessions.md
# Heading  : Lower-level concurrency with gather()
# ID       : lazy_wiki/human/sessions.md::lower-level-concurrency-with-gather::00
# Kind     : async_code
# Testable : full_exec

import asyncio
from lazybridge import LazyAgent, LazySession

sess = LazySession()
agent_a = LazyAgent("anthropic", name="news_a", session=sess)
agent_b = LazyAgent("openai",    name="news_b", session=sess)
agent_c = LazyAgent("google",    name="news_c", session=sess)

async def run():
    results = await sess.gather(
        agent_a.aloop("Summarise AI news from the US this week"),
        agent_b.aloop("Summarise AI news from Europe this week"),
        agent_c.aloop("Summarise AI news from Asia this week"),
    )
    # results[i] are CompletionResponse objects — full access to usage, tool_calls, etc.
    for r in results:
        print(r.content[:200])
        print(f"  tokens: {r.usage.input_tokens}in / {r.usage.output_tokens}out")

asyncio.run(run())
