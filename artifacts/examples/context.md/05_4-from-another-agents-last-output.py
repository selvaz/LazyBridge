# Source   : lazy_wiki/human/context.md
# Heading  : 4. From another agent's last output
# ID       : lazy_wiki/human/context.md::4-from-another-agents-last-output::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyAgent, LazyContext

researcher = LazyAgent("anthropic", name="researcher")
ctx = LazyContext.from_agent(researcher)
