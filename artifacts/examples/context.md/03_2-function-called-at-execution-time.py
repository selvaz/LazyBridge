# Source   : lazy_wiki/human/context.md
# Heading  : 2. Function (called at execution time)
# ID       : lazy_wiki/human/context.md::2-function-called-at-execution-time::00
# Kind     : context
# Testable : smoke_exec

from lazybridge import LazyContext

from datetime import date

def current_date() -> str:
    return f"Today is {date.today().isoformat()}"

ctx = LazyContext.from_function(current_date)
