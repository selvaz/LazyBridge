# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw Anthropic SDK
# ID       : lazy_wiki/human/comparison.md::raw-anthropic-sdk::03
# Kind     : local
# Testable : local_exec

import anthropic

client = anthropic.Anthropic()
resp = client.beta.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    betas=["web-search-2026-03-05"],          # beta header required — changes without notice
    tools=[{"type": "web_search_20260209"}],   # provider-specific tool type string
    messages=[{"role": "user", "content": "What happened in AI this week?"}],
)
# Filter content blocks — response mixes TextBlock and ToolResultBlock
text    = next(b.text for b in resp.content if hasattr(b, "text"))
sources = [b for b in resp.content if b.type == "web_search_result"]
print(text)
for s in sources:
    print(s.url, s.title)
