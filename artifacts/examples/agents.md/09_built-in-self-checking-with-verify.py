# Source   : lazy_wiki/human/agents.md
# Heading  : Built-in self-checking with verify=
# ID       : lazy_wiki/human/agents.md::built-in-self-checking-with-verify::00
# Kind     : verify
# Testable : smoke_exec

result = ai.loop(
    "Write a 200-word summary of transformer architecture.",
    tools=[search],
    verify="Check the summary is accurate, self-contained, and exactly 200 words. "
           "Reply with PASS or FAIL and a reason.",
    max_verify=2,   # retry up to 2 times (default: 1)
)
print(result.content)
