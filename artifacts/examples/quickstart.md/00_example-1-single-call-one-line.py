# Source   : lazy_wiki/human/quickstart.md
# Heading  : Example 1 — Single call (one line)
# ID       : lazy_wiki/human/quickstart.md::example-1-single-call-one-line::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

ai = LazyAgent("anthropic")

# Simple text response
answer = ai.text("What is the capital of France?")
print(answer)  # "Paris"

# Full response object (tokens, stop reason, etc.)
resp = ai.chat("Explain quantum entanglement in one sentence.")
print(resp.content)
print(f"Tokens used: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")
