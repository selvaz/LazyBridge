# Source   : lazy_wiki/human/agents.md
# Heading  : Thinking (extended reasoning)
# ID       : lazy_wiki/human/agents.md::thinking-extended-reasoning::01
# Kind     : llm_chat
# Testable : full_exec

from lazybridge.core.types import ThinkingConfig

resp = ai.chat(
    "Design a distributed caching system",
    thinking=ThinkingConfig(enabled=True, effort="high", budget_tokens=8000),
)
