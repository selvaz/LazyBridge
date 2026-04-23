# HumanEngine

## Example

```python
from lazybridge import Agent, HumanEngine
from pydantic import BaseModel

class Review(BaseModel):
    approved: bool
    comment: str
    rating: int

reviewer = Agent(
    engine=HumanEngine(timeout=120, default="no comment"),
    output=Review,
    name="reviewer",
)

# In a pipeline: draft → review → finalise.
pipeline = Agent.chain(drafter, reviewer, finaliser)
pipeline("draft the release notes")
```

## Pitfalls

- The terminal UI blocks the current process. In a web app, supply a
  custom ``ui=`` adapter implementing ``prompt(task, *, tools,
  output_type) -> str``.
- ``timeout=`` uses the event loop, not signals; it works in async
  contexts but may hang in tightly-blocking sync nests.

!!! note "API reference"

    HumanEngine(
        *,
        ui: Literal["terminal", "web"] | _UIProtocol = "terminal",
        timeout: float | None = None,
        default: str | None = None,
    ) -> Engine
    
    Usage: Agent(engine=HumanEngine(), tools=[...], output=Pydantic)
    
    # When output= is a Pydantic model, the terminal UI prompts field-by-field.

!!! warning "Rules & invariants"

    - ``HumanEngine`` prompts the human for input and returns it as an
      Envelope. It implements the same ``Engine`` protocol as ``LLMEngine``,
      so ``Agent(engine=HumanEngine())`` is a drop-in replacement.
    - ``output=SomeModel`` switches to per-field prompting (terminal UI).
    - ``timeout`` triggers ``default`` if set, else raises ``TimeoutError``.
    - Tool invocation is NOT handled by HumanEngine — the human types a
      raw string, they don't call tools interactively. If you want the
      human to call tools, use ``SupervisorEngine``.

