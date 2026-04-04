# Source   : lazy_wiki/human/comparison.md
# Heading  : Raw Anthropic SDK
# ID       : lazy_wiki/human/comparison.md::raw-anthropic-sdk::01
# Kind     : local
# Testable : local_exec

import json
import anthropic
from pydantic import BaseModel, ValidationError

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

client = anthropic.Anthropic()

for attempt in range(3):   # manual retry loop
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": "Generate article ideas about AI safety. Return JSON."
        }],
    )
    raw = resp.content[0].text
    try:
        data = json.loads(raw)
        article = Article(**data)
        print(article.title)
        break
    except (json.JSONDecodeError, ValidationError) as e:
        if attempt == 2:
            raise
        # retry
