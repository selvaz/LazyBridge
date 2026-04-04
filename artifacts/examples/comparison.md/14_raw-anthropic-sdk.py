# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw Anthropic SDK
# ID       : lazy_wiki/human/comparison.md::raw-anthropic-sdk::04
# Kind     : local
# Testable : local_exec

import anthropic

client = anthropic.Anthropic()
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=512,
    messages=[{"role": "user", "content": "Write a haiku about Python."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
print()
