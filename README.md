# LazyBridge

Zero-boilerplate multi-provider LLM agent framework. One class for every LLM interaction, automatic tool schema generation, composable context injection, and serializable multi-agent pipelines.

## Quick start

```python
from lazybridge import LazyAgent

ai = LazyAgent("anthropic")
print(ai.text("What is the capital of France?"))
```

Same code on any provider — change one string:

```python
LazyAgent("openai")
LazyAgent("google")
LazyAgent("deepseek")
```

## Tool loop

```python
from lazybridge import LazyAgent, LazyTool

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"{city}: 22°C, sunny"

result = LazyAgent("anthropic").loop(
    "What's the weather in Rome and Paris?",
    tools=[LazyTool.from_function(get_weather)],
)
print(result.content)
```

Schema generated automatically from type hints and docstring. No JSON dict, no decorator boilerplate.

## Conversational memory

```python
from lazybridge import LazyAgent, Memory

ai  = LazyAgent("anthropic")
mem = Memory()

ai.chat("My name is Marco", memory=mem)
resp = ai.chat("What's my name?", memory=mem)
print(resp.content)   # "Marco"
```

## Structured output

```python
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    tags: list[str]

article = LazyAgent("openai").json("Summarise AI in 2025", Article)
print(article.title)
```

## Multi-agent pipeline

```python
from lazybridge import LazyAgent, LazySession, LazyContext, LazyTool

sess       = LazySession()
researcher = LazyAgent("anthropic", name="researcher", session=sess)
writer     = LazyAgent("openai",    name="writer",     session=sess)

search_tool = LazyTool.from_function(lambda query: f"Papers about {query}")
researcher.loop("Find top 3 AI papers this week", tools=[search_tool])
result = writer.chat(
    "Write a blog post",
    context=LazyContext.from_agent(researcher),
)
print(result.content)
print(sess.graph.to_json())   # serializable pipeline topology for GUI
```

## Native provider tools (web search, code execution, …)

```python
from lazybridge.core.types import NativeTool

resp = ai.chat(
    "What happened in AI this week?",
    native_tools=[NativeTool.WEB_SEARCH],
)
for src in resp.grounding_sources:
    print(src.url, src.title)
```

## Supported providers

| Provider | String | Default model |
|---|---|---|
| Anthropic | `"anthropic"` / `"claude"` | claude-sonnet-4-6 |
| OpenAI | `"openai"` / `"gpt"` | gpt-5.4 |
| Google | `"google"` / `"gemini"` | gemini-2.5-flash |
| DeepSeek | `"deepseek"` | deepseek-chat |

## Installation

```bash
pip install lazybridge

# Provider extras (choose what you need)
pip install lazybridge[anthropic]   # Anthropic / Claude
pip install lazybridge[openai]      # OpenAI / GPT
pip install lazybridge[google]      # Google / Gemini
pip install lazybridge[all]         # all providers
```

## Project structure

```
LazyBridge/
├── lazybridge/      # Main package
│   ├── lazy_agent.py         # LazyAgent — single entry point for LLM calls
│   ├── lazy_session.py       # LazySession — shared store, events, graph
│   ├── lazy_tool.py          # LazyTool — tool schema + execution
│   ├── lazy_context.py       # LazyContext — composable system prompt injection
│   ├── lazy_store.py         # LazyStore — flat key-value blackboard (SQLite or in-memory)
│   ├── lazy_router.py        # LazyRouter — conditional branching node
│   ├── memory.py             # Memory — stateful conversation history
│   ├── graph/                # GraphSchema — serializable pipeline topology
│   └── core/                 # Provider adapters, executor, tool schema builder
└── lazy_wiki/
    ├── bot/                  # LLM-optimised reference (exhaustive, structured)
    └── human/                # Human-readable guides and SDK comparison
```

## Documentation

| Audience | Entry point |
|---|---|
| Developer | [`lazy_wiki/human/quickstart.md`](lazy_wiki/human/quickstart.md) |
| SDK comparison | [`lazy_wiki/human/comparison.md`](lazy_wiki/human/comparison.md) |
| LLM / AI assistant | [`lazy_wiki/bot/INDEX.md`](lazy_wiki/bot/INDEX.md) |
| Full API reference | [`lazy_wiki/bot/00_quickref.md`](lazy_wiki/bot/00_quickref.md) |

## License

[MIT](LICENSE)
