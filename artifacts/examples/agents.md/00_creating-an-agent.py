# Source   : lazy_wiki/human/agents.md
# Heading  : Creating an agent
# ID       : lazy_wiki/human/agents.md::creating-an-agent::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

# Minimal — just pick a provider
ai = LazyAgent("anthropic")

# With options
ai = LazyAgent(
    "anthropic",
    name="my_assistant",           # used in logs and when exposing as a tool
    model="claude-sonnet-4-6",     # override provider default
    system="You are a terse assistant. Answer in one sentence.",
    max_retries=3,                 # retry on rate limits / server errors
)
