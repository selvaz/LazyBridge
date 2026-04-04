# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

answer = LazyAgent("anthropic").text("What is the capital of France?")
print(answer)
