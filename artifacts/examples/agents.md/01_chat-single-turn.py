# Source   : lazy_wiki/human/agents.md
# Heading  : chat() — single turn
# ID       : lazy_wiki/human/agents.md::chat-single-turn::00
# Kind     : llm_chat
# Testable : smoke_exec

resp = ai.chat("What is 2 + 2?")
print(resp.content)       # "4"
print(resp.usage.input_tokens)
print(resp.stop_reason)   # "end_turn"
