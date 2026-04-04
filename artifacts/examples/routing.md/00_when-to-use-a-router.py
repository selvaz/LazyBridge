# Source   : lazy_wiki/human/routing.md
# Heading  : When to use a router
# ID       : lazy_wiki/human/routing.md::when-to-use-a-router::00
# Kind     : llm_chat
# Testable : smoke_exec

result = checker.chat("evaluate this draft")
if "approved" in result.content.lower():
    writer.chat("publish this: " + result.content)
else:
    reviewer.chat("revise this: " + result.content)
