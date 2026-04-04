# Source   : lazy_wiki/human/routing.md
# Heading  : Basic usage
# ID       : lazy_wiki/human/routing.md::basic-usage::00
# Kind     : advanced
# Testable : full_exec

from lazybridge import LazyAgent, LazyRouter

writer   = LazyAgent("anthropic", name="writer")
reviewer = LazyAgent("openai",    name="reviewer")

router = LazyRouter(
    condition=lambda response: "writer" if "APPROVED" in response.upper() else "reviewer",
    routes={"writer": writer, "reviewer": reviewer},
    name="approval_gate",
)

checker = LazyAgent("anthropic", name="checker")
result  = checker.chat("Evaluate this draft: [...]")

next_agent = router.route(result.content)
final = next_agent.chat("Proceed with: " + result.content)
print(final.content)
