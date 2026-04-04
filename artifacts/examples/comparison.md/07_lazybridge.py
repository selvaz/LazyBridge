# Source   : lazy_wiki/human/comparison.md
# Heading  : LazyBridge
# ID       : lazy_wiki/human/comparison.md::lazybridge::03
# Kind     : structured_output
# Testable : smoke_exec

import json

from lazybridge import LazyAgent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

article = LazyAgent("anthropic").json("Generate article ideas about AI safety.", Article)
print(article.title)
