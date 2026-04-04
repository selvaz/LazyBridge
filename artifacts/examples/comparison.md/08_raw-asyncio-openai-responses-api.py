# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw asyncio + OpenAI (Responses API)
# ID       : lazy_wiki/human/comparison.md::raw-asyncio-openai-responses-api::00
# Kind     : async_code
# Testable : full_exec

import asyncio
import openai

client = openai.AsyncOpenAI()

async def run_agent(task: str) -> str:
    resp = await client.responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": task}]
    )
    # Responses API: iterate output items to extract text
    message_item = next(item for item in resp.output if item.type == "message")
    return message_item.content[0].text

async def main():
    tasks = [
        "Summarise AI news from the US",
        "Summarise AI news from Europe",
        "Summarise AI news from Asia",
    ]
    results = await asyncio.gather(*[run_agent(t) for t in tasks])
    combined = "\n\n".join(results)

    summary_resp = await client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": combined},
            {"role": "user", "content": "Write a global digest"},
        ]
    )
    message_item = next(item for item in summary_resp.output if item.type == "message")
    print(message_item.content[0].text)

asyncio.run(main())
