# LazyBridge Course — Zero to Hero

A hands-on course that takes you from your first LLM call to production multi-agent pipelines with guardrails and evals.

## Prerequisites

- Python 3.11+
- Basic Python knowledge (functions, classes, type hints)
- An API key for at least one provider (Anthropic, OpenAI, or Google)

## Modules

| # | Module | What you'll learn | Time |
|---|--------|-------------------|------|
| 1 | [Your First Agent](01-first-agent.md) | Install, configure, make your first LLM call | 15 min |
| 2 | [Tools & Functions](02-tools.md) | Give agents the ability to call your code | 25 min |
| 3 | [Structured Output](03-structured-output.md) | Get typed Pydantic objects back from LLMs | 20 min |
| 4 | [Memory & Conversations](04-memory.md) | Build stateful chat conversations | 15 min |
| 5 | [Context Injection](05-context.md) | Compose context from multiple sources | 20 min |
| 6 | [Multi-Agent Pipelines](06-pipelines.md) | Chain and parallelize agents | 30 min |
| 7 | [Streaming](07-streaming.md) | Real-time token-by-token output | 15 min |
| 8 | [Guardrails & Safety](08-guardrails.md) | Validate inputs/outputs, block harmful content | 25 min |
| 9 | [Observability & Cost](09-observability.md) | Track tokens, costs, and export to OpenTelemetry | 20 min |
| 10 | [Evals & Testing](10-evals.md) | Systematically test agent output quality | 20 min |
| 11 | [Advanced & Production](11-advanced.md) | Custom providers, thinking, verify loops, tool persistence, full production pipeline | 40 min |

## How to use this course

Each module is self-contained. Code examples are runnable — copy-paste them into a Python file and run. Each module builds on concepts from previous ones, so going in order is recommended for beginners.

**Convention:** Examples use `"anthropic"` as the provider. Replace with `"openai"` or `"google"` if you prefer — the code is identical.

## Quick setup

```bash
pip install lazybridge[anthropic]
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then start with [Module 1: Your First Agent](01-first-agent.md).
