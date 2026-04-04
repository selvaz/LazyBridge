# Source   : lazy_wiki/human/routing.md
# Heading  : Multiple routes
# ID       : lazy_wiki/human/routing.md::multiple-routes::00
# Kind     : advanced
# Testable : full_exec

from lazybridge import LazyAgent, LazyRouter

researcher = LazyAgent("anthropic", name="researcher")
analyst    = LazyAgent("openai",    name="analyst")
writer     = LazyAgent("anthropic", name="writer")
reviewer   = LazyAgent("openai",    name="reviewer")

def pick_route(response: str) -> str:
    response_lower = response.strip().lower()
    if "research" in response_lower:
        return "research"
    elif "analyse" in response_lower or "analyze" in response_lower:
        return "analyse"
    elif "write" in response_lower:
        return "write"
    else:
        return "review"   # default

router = LazyRouter(
    condition=pick_route,
    routes={
        "research": researcher,
        "analyse":  analyst,
        "write":    writer,
        "review":   reviewer,
    },
    name="task_router",
    default="review",   # fallback if condition returns unknown key
)
