# Source   : lazy_wiki/human/context.md
# Heading  : 3. From the shared store
# ID       : lazy_wiki/human/context.md::3-from-the-shared-store::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyContext

from lazybridge import LazyStore

store = LazyStore()

# Only specific keys:
ctx = LazyContext.from_store(store, keys=["research", "style_guide"])

# All keys:
ctx = LazyContext.from_store(store)
