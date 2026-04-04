# Source   : lazy_wiki/human/agents.md
# Heading  : text() and json() shortcuts
# ID       : lazy_wiki/human/agents.md::text-and-json-shortcuts::00
# Kind     : structured_output
# Testable : smoke_exec

import json

# Returns str directly (no CompletionResponse wrapper)
answer = ai.text("What is the speed of light?")

# Returns a typed object (Pydantic model or dict)
from pydantic import BaseModel

class Summary(BaseModel):
    headline: str
    bullets: list[str]

summary = ai.json("Summarise AI in 2024", Summary)
print(summary.headline)
print(summary.bullets)
