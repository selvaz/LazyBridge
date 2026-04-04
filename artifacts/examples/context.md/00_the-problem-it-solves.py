# Source   : lazy_wiki/human/context.md
# Heading  : The problem it solves
# ID       : lazy_wiki/human/context.md::the-problem-it-solves::00
# Kind     : local
# Testable : local_exec

from datetime import date, datetime

# The old way — fragile, hard to compose
system = f"""
You are a writer.
Here is the research: {researcher_output}
Style guide: {style_guide}
Current date: {datetime.now()}
"""
