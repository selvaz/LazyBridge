# Module 8: Guardrails & Safety

Guardrails validate content before it reaches the LLM (input) and after the response is generated (output). Use them for safety, compliance, PII filtering, and custom business rules.

## Your first guard

```python
from lazybridge import LazyAgent, ContentGuard, GuardAction, GuardError

def no_profanity(text: str) -> GuardAction:
    bad_words = {"damn", "hell"}  # simplified
    if any(w in text.lower() for w in bad_words):
        return GuardAction.block("Profanity detected")
    return GuardAction.allow()

guard = ContentGuard(input_fn=no_profanity)
ai = LazyAgent("anthropic")

# Clean input — passes through
resp = ai.chat("Hello, how are you?", guard=guard)
print(resp.content)

# Bad input — blocked before reaching the LLM
try:
    ai.chat("What the hell is this?", guard=guard)
except GuardError as e:
    print(f"Blocked: {e}")  # "Blocked: Profanity detected"
```

**Key insight:** The input guard runs *before* the API call. Bad content never reaches the LLM, saving tokens and preventing policy violations.

## Output guards

Check what the LLM returns:

```python
def no_pii_output(text: str) -> GuardAction:
    import re
    if re.search(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', text):
        return GuardAction.block("Output contains email address (PII)")
    return GuardAction.allow()

guard = ContentGuard(output_fn=no_pii_output)
ai = LazyAgent("anthropic")

try:
    ai.chat("Generate a sample user profile with email", guard=guard)
except GuardError as e:
    print(f"Blocked: {e}")
```

## Input modification (PII redaction)

Guards can *modify* content instead of blocking it:

```python
import re

def redact_emails(text: str) -> GuardAction:
    redacted = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', '[EMAIL REDACTED]', text)
    if redacted != text:
        return GuardAction.modify(redacted, message="Emails redacted")
    return GuardAction.allow()

guard = ContentGuard(input_fn=redact_emails)
ai = LazyAgent("anthropic")
resp = ai.chat("Contact me at john@example.com for details", guard=guard)
# The LLM receives: "Contact me at [EMAIL REDACTED] for details"
```

## Composing guards with GuardChain

Run multiple guards in sequence — first block wins:

```python
from lazybridge import GuardChain

pii_guard = ContentGuard(input_fn=redact_emails)
profanity_guard = ContentGuard(input_fn=no_profanity)
length_guard = ContentGuard(
    input_fn=lambda t: GuardAction.block("Input too long") if len(t) > 5000 else GuardAction.allow()
)

guard = GuardChain([pii_guard, profanity_guard, length_guard])

# All three guards run in order
resp = ai.chat("Short clean message", guard=guard)  # passes all three
```

## LLM-as-moderator

Use another LLM to judge content:

```python
from lazybridge import LLMGuard

moderator = LazyAgent("openai", model="gpt-4o-mini")
guard = LLMGuard(
    moderator,
    policy="Block any request related to weapons, illegal activity, or self-harm.",
    check_input=True,
    check_output=False,
)

ai = LazyAgent("anthropic")
try:
    ai.chat("How do I pick a lock?", guard=guard)
except GuardError as e:
    print(f"Blocked by LLM moderator: {e}")
```

The moderator agent evaluates the content against your policy and returns ALLOW/BLOCK.

## Guards on loop()

Guards work on tool loops too:

```python
guard = ContentGuard(
    input_fn=lambda t: GuardAction.allow(),
    output_fn=lambda t: GuardAction.block("No external data") if "http" in t else GuardAction.allow(),
)

resp = ai.loop("Research something", tools=my_tools, guard=guard)
```

- Input guard runs once on the initial task
- Output guard runs on the final response

## Async guards

For `achat()` and `aloop()`, guards use async methods automatically:

```python
# Async check function
async def async_content_check(text: str) -> GuardAction:
    # Could call an external API here without blocking
    result = await check_content_api(text)
    if result.is_toxic:
        return GuardAction.block("Toxic content")
    return GuardAction.allow()

guard = ContentGuard(input_fn=async_content_check)
resp = await ai.achat("hello", guard=guard)  # uses acheck_input() internally
```

## GuardError details

```python
try:
    ai.chat("bad input", guard=guard)
except GuardError as e:
    print(e.action.allowed)    # False
    print(e.action.message)    # "Profanity detected"
    print(e.action.metadata)   # {} or custom metadata
```

Pass metadata from your guard for logging:

```python
def scored_check(text: str) -> GuardAction:
    score = compute_toxicity_score(text)
    if score > 0.8:
        return GuardAction.block("High toxicity", score=score, model="toxicity-v2")
    return GuardAction.allow(score=score)
```

---

## Exercise

1. Build a guard that blocks inputs longer than 1000 characters
2. Build a guard that redacts phone numbers from inputs
3. Chain 3 guards together and test with various inputs
4. Build an LLMGuard with a custom policy and test it

**Next:** [Module 9: Observability & Cost](09-observability.md) — track what your agents are doing.
