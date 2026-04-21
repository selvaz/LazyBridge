# HumanEngine

`HumanEngine` is the simplest form of human-in-the-loop: the engine
pauses execution and waits for the human to type something. That's it.
No REPL, no tool calling, no agent retry — just a text prompt and a
response.

Use it when the human is a participant in a pipeline (an approver, a
reviewer, an annotator) rather than an operator. The Agent wrapping
`HumanEngine` slots identically into `chain` / `parallel` / `Plan`.

When `output=` is a Pydantic model, the terminal variant prompts
field-by-field with coercion — integers become `int`, `bool` accepts
`yes/no`, lists are comma-separated. That's enough for light-touch
forms without shipping a GUI.

For a full interactive REPL with tool calls and agent retry, use
`SupervisorEngine` instead.

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

## See also

[supervisor](supervisor.md), [agent](agent.md),
decision tree: [human_engine_vs_supervisor](../decisions/human-engine-vs-supervisor.md)
