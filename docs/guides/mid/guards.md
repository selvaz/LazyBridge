# Guards

Hard input / output filters that run **before** and **after** the engine.
A blocked input never reaches the engine; a blocked output is replaced
with an error envelope. Compose cheap deterministic guards with an
LLM-as-judge fallback for what regex can't see.

## Signature

```python
from lazybridge import Guard, ContentGuard, GuardChain, LLMGuard, GuardAction


# The protocol every guard satisfies.
class Guard:
    async def acheck_input(self, text: str) -> GuardAction
    async def acheck_output(self, text: str) -> GuardAction


# Verdict object.
GuardAction(
    allowed=True,                 # False blocks the run
    message=None,                 # error message when blocked
    modified_text=None,           # rewrite the input or output text
    metadata={},                  # opaque dict carried into the event log
)


# Built-ins.
ContentGuard(
    input_fn=None,                # callable(text) -> GuardAction (input gate)
    output_fn=None,               # callable(text) -> GuardAction (output gate)
)

GuardChain(*guards)               # first blocker wins; short-circuits on allowed=False

LLMGuard(
    judge,                        # an Agent — its run() decides
    policy,                       # str describing what to allow / reject
    *,
    timeout=60.0,                 # deadline for the judge; None = unbounded
)


class GuardError(Exception):      # raised by some integrations on hard policy failure
    ...
```

Pass a guard to an `Agent` via `guard=...`. To stack multiple, wrap them
in a `GuardChain`.

## Synopsis

A guard is a hard gate. `acheck_input` runs **before** the engine — if
it returns `allowed=False`, the engine is never invoked and the agent
returns an error envelope. `acheck_output` runs **after** the engine
on `Envelope.text()` — if it blocks, the payload is replaced with an
error envelope (type `GuardBlocked`).

Either gate can also **rewrite** instead of blocking: returning
`GuardAction(allowed=True, modified_text="…")` from `acheck_input`
replaces the engine's task; the same on `acheck_output` replaces the
payload string.

`GuardChain` runs guards in order and short-circuits on the first
`allowed=False`. The convention is **cheap first, LLM last**: a regex
or substring `ContentGuard` runs in microseconds, an `LLMGuard` only
fires when the cheap layer didn't decide. Saves tokens.

`Agent.stream()` enforces guards too — `acheck_input` runs before the
first token; a blocked task raises `ValueError` instead of silently
streaming.

## When to use it

- **Compliance / safety policies** that must hold regardless of the
  engine's behaviour. The agent literally can't bypass a guard
  because it never sees blocked inputs.
- **Layered defence in depth.** Pair a deterministic regex guard
  (cheap, fast, false-positive-prone) with an `LLMGuard` (slow, more
  nuanced) so the LLM only adjudicates the hard cases.
- **Output redaction.** Use a `ContentGuard(output_fn=...)` that
  returns `GuardAction(allowed=True, modified_text=redacted)` to mask
  PII before the user ever sees the payload.
- **Streaming workflows where you must enforce at first byte.**
  `acheck_input` runs synchronously before streaming begins.

## When NOT to use it

- **Soft preferences ("the model should generally avoid X").** Guards
  are hard gates with no feedback loop — once blocked, the run ends.
  For "try again with feedback" semantics, use `verify=` (Phase 3).
- **Conversation-level rules that need history context.** A guard
  sees only the current task / output text, not memory. Use a
  `verify=` agent or a custom engine wrapper for stateful policies.
- **Performance-critical paths where every microsecond counts.** A
  regex `ContentGuard` is essentially free; an `LLMGuard` adds a
  judge call. Profile before stacking too many layers.

## Example

```python
import re

from lazybridge import (
    Agent,
    ContentGuard,
    GuardAction,
    GuardChain,
    LLMEngine,
    LLMGuard,
)


# 1) Cheap regex guard — block input mentioning email addresses.
def no_emails(text: str) -> GuardAction:
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return GuardAction(
            allowed=False,
            message="Remove email addresses before submitting.",
        )
    return GuardAction(allowed=True)


# 2) LLM-as-judge for harder policy violations.
judge = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system='Respond "approved" or "rejected: <reason>".',
    ),
    name="judge",
)


# 3) Compose: cheap first, LLM last.
guard = GuardChain(
    ContentGuard(input_fn=no_emails),
    LLMGuard(
        judge,
        policy="Reject outputs that contain medical advice.",
        timeout=10.0,
    ),
)


bot = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    guard=guard,
    name="bot",
)


# Blocked by the cheap regex — engine is never invoked.
result = bot("my email is foo@bar.com, what's the weather?")
assert not result.ok
print(result.error.type, result.error.message)


# 4) Output rewrite — redact PII before the caller sees the payload.
def redact_phone_numbers(text: str) -> GuardAction:
    redacted = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED]", text)
    if redacted != text:
        return GuardAction(allowed=True, modified_text=redacted)
    return GuardAction(allowed=True)


sanitiser = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    guard=ContentGuard(output_fn=redact_phone_numbers),
)
```

## Pitfalls

- **A guard that raises aborts the run.** Always return
  `GuardAction(allowed=False, message=str(e))` on internal errors;
  letting an exception escape produces an unrecoverable failure
  instead of a structured rejection.
- **`LLMGuard.timeout` is honoured on both sync and async paths.**
  The sync path uses a daemon thread + `join(timeout=)`; the async
  path uses `asyncio.wait_for`. On timeout the guard fails closed
  (blocked). `timeout=None` restores unbounded behaviour — only do
  this if you trust your judge to be fast.
- **`LLMGuard` costs tokens on every call.** Order it last in a
  `GuardChain` so the cheap layer catches the obvious cases first.
- **Guards see `Envelope.text()`, not the typed payload.** If you're
  using structured output (`output=PydanticModel`), the output guard
  operates on the JSON serialisation. Check string content
  accordingly.
- **`modified_text` on output replaces the payload string** but does
  not re-validate against `output=`. If you redact a structured
  output's JSON, the consumer may receive invalid JSON; prefer to
  redact within the model's `model_dump()` output instead.
- **`GuardChain` short-circuits on the first `allowed=False`.**
  Subsequent guards never run, including their side-effect-free
  observations. Don't rely on a downstream guard to "also see" the
  blocked input.
- **Streaming respects guards.** `agent.stream(task)` raises
  `ValueError` on a blocked input rather than yielding silently.
  Catch it explicitly in streaming UIs.

## See also

- [Session](session.md) — guard outcomes are emitted as events
  (`metadata` from `GuardAction` is preserved on the event payload).
- [Agent](../basic/agent.md) — `guard=` is a first-class kwarg
  alongside `tools=`, `memory=`, `output=`.
- *Guides → Full → verify=* (Phase 3) — different placement: a
  judge wraps the agent's *output* with a retry feedback loop,
  rather than acting as a hard gate.
