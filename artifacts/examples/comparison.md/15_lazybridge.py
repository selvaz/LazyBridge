# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::07
# Kind     : streaming
# Testable : full_exec

from lazybridge import LazyAgent

for chunk in LazyAgent("anthropic").chat("Write a haiku about Python.", stream=True):
    print(chunk.delta, end="", flush=True)
print()
