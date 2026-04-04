# Source   : lazy_wiki/human/context.md
# Heading  : Testing your context
# ID       : lazy_wiki/human/context.md::testing-your-context::00
# Kind     : llm_loop
# Testable : smoke_exec

# Before running any agents:
print(ctx())   # only static parts appear

# After running the researcher:
researcher.loop("find data")
print(ctx())   # now includes researcher's output
