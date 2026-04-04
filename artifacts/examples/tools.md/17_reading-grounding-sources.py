# Source   : lazy_wiki/human/tools.md
# Heading  : Reading grounding sources
# ID       : lazy_wiki/human/tools.md::reading-grounding-sources::00
# Kind     : native_tools
# Testable : full_exec

from lazybridge.core.types import NativeTool

resp = ai.chat("Who won the last Formula 1 championship?", native_tools=[NativeTool.WEB_SEARCH])

for src in resp.grounding_sources:
    print(src.url)
    print(src.title)
    print(src.snippet)

# Google also exposes the actual queries issued
print(resp.web_search_queries)   # e.g. ["F1 2025 champion", "Formula 1 championship winner"]
