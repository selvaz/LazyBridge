# Source   : lazy_wiki/human/routing.md
# Heading  : Fallback default
# ID       : lazy_wiki/human/routing.md::fallback-default::00
# Kind     : advanced
# Testable : full_exec

from lazybridge import LazyRouter

router = LazyRouter(
    condition=lambda r: r.strip().lower(),
    routes={"approve": publisher, "reject": drafter},
    name="publish_gate",
    default="reject",   # any unknown response → drafter
)
