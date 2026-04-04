# Source   : lazy_wiki/human/routing.md
# Heading  : Async condition
# ID       : lazy_wiki/human/routing.md::async-condition::00
# Kind     : async_code
# Testable : full_exec

import asyncio
from lazybridge import LazyAgent, LazyRouter

# Create once — reused for every routing decision
classifier = LazyAgent("anthropic")

async def classify_with_llm(text: str) -> str:
    label = await classifier.atext(
        f"Classify this task as one of: research / analyse / write. Task: {text}. Return only the label."
    )
    return label.strip().lower()

router = LazyRouter(
    condition=classify_with_llm,
    routes={"research": researcher, "analyse": analyst, "write": writer},
    default="write",
)

next_agent = asyncio.run(router.aroute("What are the latest GPU benchmark numbers?"))
