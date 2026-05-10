"""Single agent — the canonical first-contact shape.

```python
from lazybridge import Agent, LLMEngine
agent = Agent(engine=LLMEngine("claude-opus-4-7"))
result = agent("hello world")
result.text()
```

Replace ``MockAgent`` with ``LLMEngine`` to call a real provider.
"""

from __future__ import annotations

from lazybridge.testing import MockAgent


def main() -> None:
    agent = MockAgent(["Bonjour!"], name="greeter")
    result = agent("Translate 'hello' to French.")
    print(result.text())
    # Envelope metadata rolls up cost/tokens/latency across nested calls.
    print(f"cost=${result.metadata.cost_usd}; input_tokens={result.metadata.input_tokens}")


if __name__ == "__main__":
    main()
