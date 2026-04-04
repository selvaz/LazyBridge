# Source   : lazy_wiki/human/agents.md
# Heading  : Async
# ID       : lazy_wiki/human/agents.md::async::00
# Kind     : structured_output
# Testable : smoke_exec

import asyncio

async def main():
    resp   = await ai.achat("Hello")
    result = await ai.aloop("Find news", tools=[...])
    text   = await ai.atext("Hello")
    data   = await ai.ajson("...", MyModel)

asyncio.run(main())
