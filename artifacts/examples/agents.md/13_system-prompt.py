# Source   : lazy_wiki/human/agents.md
# Heading  : System prompt
# ID       : lazy_wiki/human/agents.md::system-prompt::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

ai = LazyAgent(
    "anthropic",
    system="You are a Python expert. Only use Python code in your answers.",
)

# Add a per-call addition (appended to the agent-level system)
resp = ai.chat("Show me a list", system="Use bullet points.")
