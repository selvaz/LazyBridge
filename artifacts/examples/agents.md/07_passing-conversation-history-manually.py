# Source   : lazy_wiki/human/agents.md
# Heading  : Passing conversation history manually
# ID       : lazy_wiki/human/agents.md::passing-conversation-history-manually::01
# Kind     : llm_chat
# Testable : smoke_exec

history = [
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello Alice!"},
    {"role": "user",      "content": "What's my name?"},
]
resp = ai.chat(history)
