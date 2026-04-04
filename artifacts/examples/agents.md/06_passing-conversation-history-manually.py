# Source   : lazy_wiki/human/agents.md
# Heading  : Passing conversation history manually
# ID       : lazy_wiki/human/agents.md::passing-conversation-history-manually::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge.core.types import Message, Role

history = [
    Message(role=Role.USER,      content="My name is Alice."),
    Message(role=Role.ASSISTANT, content="Hello Alice!"),
]
resp = ai.chat(history + [Message(role=Role.USER, content="What's my name?")])
print(resp.content)  # "Alice"
