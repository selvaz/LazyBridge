# Source   : lazy_wiki/human/agents.md
# Heading  : Streaming
# ID       : lazy_wiki/human/agents.md::streaming::00
# Kind     : streaming
# Testable : full_exec

for chunk in ai.chat("Write me a haiku about Python.", stream=True):
    print(chunk.delta, end="", flush=True)
print()  # newline at end
