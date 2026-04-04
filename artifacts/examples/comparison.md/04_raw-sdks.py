# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw SDKs
# ID       : lazy_wiki/human/comparison.md::raw-sdks::00
# Kind     : local
# Testable : local_exec

import anthropic
import openai

# Two separate clients, two completely different APIs
anthropic_client = anthropic.Anthropic()
openai_client    = openai.OpenAI()

# Researcher (Anthropic — Messages API)
research_resp = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Find AI news this week"}]
)
research_output = research_resp.content[0].text  # Anthropic: .content[0].text

# Writer (OpenAI — Responses API, new default)
writer_resp = openai_client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "system", "content": f"Use this research:\n{research_output}"},
        {"role": "user",   "content": "Write a newsletter section"},
    ]
)
# OpenAI Responses API: iterate output items, find message, extract text block
message_item = next(item for item in writer_resp.output if item.type == "message")
print(message_item.content[0].text)
