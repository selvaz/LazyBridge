# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw Anthropic SDK
# ID       : lazy_wiki/human/comparison.md::raw-anthropic-sdk::02
# Kind     : local
# Testable : local_exec

import anthropic

client   = anthropic.Anthropic()
messages = []   # you own this list — you must append every turn correctly

def chat(user_msg: str) -> str:
    messages.append({"role": "user", "content": user_msg})
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=messages,
    )
    assistant_msg = resp.content[0].text
    messages.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg

print(chat("My name is Marco"))
print(chat("What is my name?"))
print(chat("What did we discuss so far?"))
