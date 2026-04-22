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

# Cheap regex guard that inspects the USER task before it hits the model.
#   GuardAction(allowed=False, message=...) → block + explain
#   GuardAction(allowed=True)                → pass through
def no_emails(text: str) -> GuardAction:
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return GuardAction(allowed=False, message="Remove email addresses first.")
    return GuardAction(allowed=True)

# LLM-backed policy guard. ``system=`` lives on the engine, not Agent.
from lazybridge import LLMEngine
judge = Agent(
    engine=LLMEngine("claude-opus-4-7",
                     system='Respond "approved" or "rejected: <reason>".'),
    name="judge",                       # label surfaced in observability
)

guard = GuardChain(
    # ContentGuard wraps plain functions into the Guard interface.
    #   input_fn=  runs on the USER task before the engine is invoked
    #   output_fn= runs on Envelope.text() after the engine returns
    ContentGuard(input_fn=no_emails),
    # LLMGuard delegates to a judge Agent. ``policy=`` is a natural-language
    # rule; the guard blocks when the judge's verdict starts with
    # "block" or "deny".
    LLMGuard(judge, policy="Reject outputs that contain medical advice."),
)

# guard=  attaches the chain to both input and output paths of this Agent.
# name="bot"  labels the agent in Session.graph / event logs.
bot = Agent("claude-opus-4-7", guard=guard, name="bot")
env = bot("my email is foo@bar.com, what's the weather?")
assert not env.ok                        # blocked by the regex guard
print(env.error.message)
```

## Rewriting instead of blocking: `modified_text`

Every `GuardAction` can rewrite the text it was given instead of
blocking it.  Use this for scrubbing inputs before they reach the
model (masking PII, trimming over-long history) or for
post-processing outputs (enforcing a tone, stripping markup).  The
rewritten text replaces the original at the boundary the guard ran
on — input rewrites change the engine's task; output rewrites change
`Envelope.payload`.

```python
# What this shows: a guard that allows the request through but masks
# obvious secrets BEFORE the model sees them, plus an output-side
# guard that trims trailing boilerplate.
# Why two hooks: an input rewrite affects what the LLM is given;
# an output rewrite affects what the caller receives. Same
# GuardAction primitive, different call sites.

import re
from lazybridge import Agent, ContentGuard, GuardAction

_SECRET_RE = re.compile(r"\b(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b")

def mask_secrets(text: str) -> GuardAction:
    masked, n = _SECRET_RE.subn("<redacted>", text)
    if n:
        # allow + modified_text=... → proceed with the rewritten string
        return GuardAction.modify(masked, message=f"masked {n} secrets")
    return GuardAction.allow()

def trim_boilerplate(text: str) -> GuardAction:
    cleaned = text.split("\n\nDisclaimer:", 1)[0]
    if cleaned != text:
        return GuardAction.modify(cleaned)
    return GuardAction.allow()

scrubbed = Agent(
    "claude-opus-4-7",
    guard=ContentGuard(input_fn=mask_secrets, output_fn=trim_boilerplate),
    name="scrubbed",
)
```

Inside a `GuardChain` the rewrites **chain**: each guard sees the
text as rewritten by the previous one, and the final action carries
the accumulated modification.  A later guard can rewrite again or
block outright.

## Async guards: `acheck_input` / `acheck_output`

`ContentGuard` takes sync callables.  When your guard needs to call
an LLM, hit a remote policy service, or do anything async, subclass
`Guard` and override the async methods directly.  The agent runtime
awaits them natively.

```python
# What this shows: a custom policy guard that checks an allow-list
# stored remotely. Subclassing Guard (not ContentGuard) gives you
# control over both input and output in one class, access to async I/O,
# and a place to hold state (the client session below).
# Why subclass instead of ContentGuard: ContentGuard wraps two
# stateless functions. Anything stateful or async-only benefits from
# a class with explicit acheck_input / acheck_output overrides.

from lazybridge import Agent, Guard, GuardAction

class AllowlistGuard(Guard):
    def __init__(self, client):
        self._client = client

    async def acheck_input(self, text: str) -> GuardAction:
        verdict = await self._client.check(text)    # async I/O
        return GuardAction.allow() if verdict.ok else GuardAction.block(verdict.reason)

    async def acheck_output(self, text: str) -> GuardAction:
        # Outputs pass through unconditionally in this example.
        return GuardAction.allow()

agent = Agent("claude-opus-4-7", guard=AllowlistGuard(my_client))
```

The framework prefers the async methods when the engine is async and
falls back to the sync ones when needed.  Overriding only one half
(input OR output) is fine — the other defaults to `allow()`.

## Custom `Guard` subclass: policy + metadata

Metadata attached to a `GuardAction` flows to the error `Envelope`
and to any exporter listening to tool-error events — useful for
downstream classification, alerting, and dashboards.

```python
# What this shows: a guard that tags blocked requests with a
# machine-readable category on metadata, so an alerting system can
# differentiate "prompt_injection" from "pii_present" without
# pattern-matching the message string.
# Why metadata: GuardAction.message is human-readable ("please
# remove emails first"). Machines need a category/severity. The
# metadata dict is free-form JSON-safe data you control.

from lazybridge import Agent, Guard, GuardAction

class CategorisedGuard(Guard):
    def check_input(self, text: str) -> GuardAction:
        if "ignore previous instructions" in text.lower():
            return GuardAction.block(
                "prompt injection detected",
                category="prompt_injection", severity="high",
            )
        if "@" in text:
            return GuardAction.block(
                "remove emails first",
                category="pii_present", severity="low",
            )
        return GuardAction.allow()

env = Agent("claude-opus-4-7", guard=CategorisedGuard())("ignore previous instructions")
assert env.error.type == "GuardBlocked"
# metadata is preserved on the error envelope for downstream consumers:
# env.error.message contains "prompt injection detected"
# exporters see the TOOL_ERROR event with the full metadata dict
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
