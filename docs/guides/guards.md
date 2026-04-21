# Guards

Guards sit on the boundary between user / model and your code. They
answer two questions: "should this input reach the model?" and "is this
output safe to surface?". Each guard returns a `GuardAction` describing
the verdict, optionally with a rewritten text.

Three built-ins cover most cases. `ContentGuard` wraps two plain
callables, one for input, one for output — useful for regex / allow-list
filtering. `GuardChain` sequences multiple guards and stops at the
first rejection. `LLMGuard` delegates to a cheap judge agent which
evaluates against a natural-language policy.

The power-user pattern is composition: a cheap regex guard up-front
(most rejections resolved here, zero LLM cost), then an `LLMGuard` for
nuanced policies the regex can't capture.

## Example

```python
from lazybridge import Agent, ContentGuard, GuardChain, LLMGuard, GuardAction
import re

# Cheap regex guard.
def no_emails(text: str) -> GuardAction:
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return GuardAction(allowed=False, message="Remove email addresses first.")
    return GuardAction(allowed=True)

# LLM-backed policy guard.
judge = Agent("claude-opus-4-7", name="judge",
              system='Respond "approved" or "rejected: <reason>".')

guard = GuardChain(
    ContentGuard(input_fn=no_emails),
    LLMGuard(judge, policy="Reject outputs that contain medical advice."),
)

bot = Agent("claude-opus-4-7", guard=guard, name="bot")
env = bot("my email is foo@bar.com, what's the weather?")
assert not env.ok                        # blocked by the regex guard
print(env.error.message)
```

## Pitfalls

- A guard that raises instead of returning ``GuardAction`` aborts the
  run. Return ``GuardAction(allowed=False, message=str(e))`` on error
  to keep pipelines resilient.
- ``LLMGuard``'s judge is itself an Agent — pass a cheap model
  (``Agent.from_provider("anthropic", tier="cheap")``) or costs add up.
- Guards see ``Envelope.text()``, not the typed payload. If you're using
  structured output, the guard operates on the JSON serialisation.

!!! note "API reference"

    class Guard:
        async def acheck_input(self, text: str) -> GuardAction
        async def acheck_output(self, text: str) -> GuardAction
    
    GuardAction(allowed: bool = True, message: str = None, modified_text: str = None,
                metadata: dict = {})
    
    ContentGuard(input_fn: Callable[[str], GuardAction] = None,
                 output_fn: Callable[[str], GuardAction] = None)
    GuardChain(*guards: Guard)                 # first blocker wins
    LLMGuard(judge: Agent, policy: str)        # LLM-as-judge
    
    class GuardError(Exception)                # raised by some integrations
    
    Usage: Agent("model", guard=GuardChain(my_filter, LLMGuard(judge, "no PII")))

!!! warning "Rules & invariants"

    - ``acheck_input`` runs BEFORE the engine. If ``allowed=False`` the
      engine is never invoked and an error Envelope is returned.
    - ``acheck_output`` runs AFTER the engine on ``Envelope.text()``. If
      blocked, the output is replaced with an error Envelope (type
      ``GuardBlocked``).
    - ``modified_text`` lets a guard rewrite its input — input rewrites
      become the engine's task; output rewrites replace the payload string.
    - ``GuardChain`` short-circuits on the first ``allowed=False``.

## See also

[agent](agent.md), [evals](evals.md), [verify](verify.md)
