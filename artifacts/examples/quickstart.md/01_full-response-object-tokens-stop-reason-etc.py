# Source   : lazy_wiki/human/quickstart.md
# Heading  : Full response object (tokens, stop reason, etc.)
# ID       : lazy_wiki/human/quickstart.md::full-response-object-tokens-stop-reason-etc::00
# Kind     : llm_chat
# Testable : smoke_exec

from lazybridge import LazyAgent

ai_openai   = LazyAgent("openai")
ai_google   = LazyAgent("google")
ai_deepseek = LazyAgent("deepseek")
