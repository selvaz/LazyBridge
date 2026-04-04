# Source   : lazy_wiki/human/agents.md
# Heading  : Reading the result
# ID       : lazy_wiki/human/agents.md::reading-the-result::00
# Kind     : structured_output
# Testable : smoke_exec

from pydantic import BaseModel
from lazybridge import LazyAgent

class Summary(BaseModel):
    headline: str
    bullets: list[str]

ai = LazyAgent("anthropic", output_schema=Summary)
ai.chat("Summarise the state of AI in 2024")

# result is a Summary instance (typed), not a string
s = ai.result
print(s.headline)
print(s.bullets)

# For agents without output_schema, result is plain text:
plain = LazyAgent("openai")
plain.chat("Hello")
print(plain.result)   # "Hello! How can I help you today?"
