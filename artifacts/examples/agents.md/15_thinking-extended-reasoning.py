# Source   : lazy_wiki/human/agents.md
# Heading  : Thinking (extended reasoning)
# ID       : lazy_wiki/human/agents.md::thinking-extended-reasoning::00
# Kind     : llm_chat
# Testable : full_exec

resp = ai.chat("What is 17 × 23?", thinking=True)
print(resp.thinking)   # internal reasoning
print(resp.content)    # final answer
