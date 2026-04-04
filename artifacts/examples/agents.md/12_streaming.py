# Source   : lazy_wiki/human/agents.md
# Heading  : Streaming
# ID       : lazy_wiki/human/agents.md::streaming::01
# Kind     : streaming
# Testable : full_exec

async for chunk in await ai.achat("Write a haiku", stream=True):
    print(chunk.delta, end="", flush=True)
