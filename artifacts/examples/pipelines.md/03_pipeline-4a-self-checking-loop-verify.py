# Source   : lazy_wiki/human/pipelines.md
# Heading  : Pipeline 4a — Self-checking Loop (verify=)
# ID       : lazy_wiki/human/pipelines.md::pipeline-4a-self-checking-loop-verify::00
# Kind     : verify
# Testable : smoke_exec

from lazybridge import LazyAgent

drafter = LazyAgent(
    "anthropic",
    system="You are a precise technical writer. Be accurate and concise.",
)

result = drafter.loop(
    "Write a 200-word intro to transformer architecture.",
    verify=(
        "Check this text: is it accurate, clearly written, and under 200 words? "
        "Reply with PASS or FAIL and a one-sentence reason."
    ),
    max_verify=3,   # retry up to 3 times before accepting as-is
)
print(result.content)
