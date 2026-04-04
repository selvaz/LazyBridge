# Source   : lazy_wiki/human/pipelines.md
# Heading  : Pipeline 4b — Multi-destination Router
# ID       : lazy_wiki/human/pipelines.md::pipeline-4b-multi-destination-router::00
# Kind     : advanced
# Testable : full_exec

from lazybridge import LazyAgent, LazyRouter, LazySession

sess = LazySession(tracking="verbose")

drafter   = LazyAgent("anthropic", name="drafter",   session=sess)
reviewer  = LazyAgent("openai",    name="reviewer",  session=sess)
publisher = LazyAgent("anthropic", name="publisher", session=sess)

router = LazyRouter(
    condition=lambda r: "publish" if "APPROVED" in r.upper() else "revise",
    routes={"publish": publisher, "revise": drafter},
    name="review_gate",
    default="revise",
)

draft = drafter.chat("Write a 200-word intro to transformer architecture.")

for revision in range(4):
    review = reviewer.chat(
        f"Review this text critically. End your response with APPROVED or REJECTED.\n\n{draft.content}"
    )
    print(f"Revision {revision + 1}: {review.content[:80]}...")

    next_agent = router.route(review.content)
    if next_agent is publisher:
        final = publisher.chat(f"Format for publication:\n\n{draft.content}")
        print("\n=== PUBLISHED ===")
        print(final.content)
        break
    else:
        draft = drafter.chat(
            f"Rewrite based on this feedback: {review.content}\n\nOriginal: {draft.content}"
        )
