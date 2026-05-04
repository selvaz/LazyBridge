# HumanEngine (ext.hil)

**Use `HumanEngine`** for an approval gate or a structured form: the
human types a string (or fills Pydantic fields), the agent treats it as
an LLM response.  Drop-in replacement for `LLMEngine` in any pipeline
where you want to insert a human at a specific step.

**Use `SupervisorEngine` instead** when the human needs to call tools,
retry agents with feedback, or run a real REPL — `HumanEngine` is the
lighter approval-only variant.

## Example

```python
from lazybridge import Agent
from lazybridge.ext.hil import HumanEngine
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

- [SupervisorEngine](supervisor.md) — full HIL REPL with tools and retry.
- [Agent.chain](chain.md) — typical pattern for inserting HumanEngine mid-pipeline.
