# LazyBridge

Zero-boilerplate multi-provider LLM agent framework.

```python
from lazybridge import LazyAgent

ai = LazyAgent("anthropic")
print(ai.text("What is the capital of France?"))
```

Same code on any provider --- change one string:

```python
LazyAgent("openai")
LazyAgent("google")
LazyAgent("deepseek")
```

## Why LazyBridge?

- **One class** for all LLM interaction --- `LazyAgent` handles chat, tool loops, streaming, structured output
- **Multi-provider** from day one --- switch between Anthropic, OpenAI, Google, DeepSeek with a string
- **Automatic tool schemas** from type hints --- no JSON dicts, no decorators
- **Composable context** via `LazyContext` --- lazy-evaluated, testable, decoupled
- **Built-in observability** --- event tracking, OpenTelemetry export, cost/token aggregation
- **Lightweight** --- only Pydantic required; provider SDKs are optional

## Supported providers

| Provider | String | Default model |
|---|---|---|
| Anthropic | `"anthropic"` / `"claude"` | claude-sonnet-4-6 |
| OpenAI | `"openai"` / `"gpt"` | gpt-5.4 |
| Google | `"google"` / `"gemini"` | gemini-3.1-pro-preview |
| DeepSeek | `"deepseek"` | deepseek-chat |

## Next steps

- [Getting Started](quickstart.md) --- installation and first steps
- [Agents](agents.md) --- chat, tools, streaming, structured output
- [Sessions & Pipelines](sessions.md) --- multi-agent orchestration
- [API Reference](reference.md) --- complete API surface
- [Troubleshooting](troubleshooting.md) --- common errors and fixes
