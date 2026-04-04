# Source   : lazy_wiki/human/routing.md
# Heading  : Full pipeline example
# ID       : lazy_wiki/human/routing.md::full-pipeline-example::00
# Kind     : advanced
# Testable : full_exec

from lazybridge import LazyAgent, LazyRouter, LazySession

sess = LazySession()

drafter   = LazyAgent("anthropic", name="drafter",   session=sess)
reviewer  = LazyAgent("openai",    name="reviewer",  session=sess)
publisher = LazyAgent("anthropic", name="publisher", session=sess)

router = LazyRouter(
    condition=lambda r: "publisher" if "APPROVED" in r.upper() else "reviewer",
    routes={"publisher": publisher, "reviewer": reviewer},
    name="quality_gate",
    default="reviewer",
)

# Pipeline loop
content = "Write a blog post about AI safety."
for _ in range(3):  # up to 3 revision cycles
    draft = drafter.chat(content)
    check = reviewer.chat(f"Review this and say APPROVED or REJECTED with reason: {draft.content}")
    next_agent = router.route(check.content)
    if next_agent is publisher:
        result = publisher.chat(f"Publish: {draft.content}")
        print("Published:", result.content[:200])
        break
    else:
        content = f"Revise based on this feedback: {check.content}\n\nOriginal draft: {draft.content}"
