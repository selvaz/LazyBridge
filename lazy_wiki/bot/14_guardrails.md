# Guardrails — Complete Reference

## 1. Overview

Guards run **before** (input) and/or **after** (output) every LLM call. They can block, silently modify, or flag content — never see each other's internals. Applied via `guard=` on `chat()`, `loop()`, `text()`, `json()`, and their async variants.

```python
from lazybridge import ContentGuard, GuardChain, LLMGuard, GuardAction, GuardError
```

---

## 2. GuardAction — the return type of every guard function

```python
@dataclass
class GuardAction:
    allowed: bool               # True = pass through; False = raise GuardError
    message: str | None         # block reason or optional note
    modified_text: str | None   # if set, replaces the original text
    metadata: dict              # arbitrary data (scores, labels, etc.)

# Constructors
GuardAction.allow()                           # pass through unchanged
GuardAction.allow(message="ok", score=0.01)   # pass with metadata
GuardAction.block("reason")                   # raise GuardError
GuardAction.block("reason", score=0.97)       # block with metadata
GuardAction.modify(new_text)                  # replace text silently
GuardAction.modify(new_text, message="Emails redacted")  # replace with note
```

`GuardAction.modify()` is `allowed=True` with `modified_text` set. The modified text is passed to the LLM (input guard) or returned as the response content (output guard).

---

## 3. GuardError

```python
class GuardError(Exception):
    action: GuardAction    # the blocking action

# Catch pattern
try:
    resp = agent.chat("...", guard=guard)
except GuardError as e:
    print(e.action.message)     # "reason for blocking"
    print(e.action.metadata)    # {"score": 0.97, ...}
```

---

## 4. Guard Protocol

Any object with `check_input` + `check_output` satisfies the `Guard` protocol. All four methods are available (sync + async):

```python
from lazybridge import Guard  # Protocol, runtime_checkable

class MyGuard:
    def check_input(self, text: str) -> GuardAction: ...
    def check_output(self, text: str) -> GuardAction: ...
    async def acheck_input(self, text: str) -> GuardAction: ...   # default: wraps sync
    async def acheck_output(self, text: str) -> GuardAction: ...  # default: wraps sync
```

---

## 5. ContentGuard — function-based

```python
ContentGuard(
    input_fn: Callable[[str], GuardAction] | None = None,
    output_fn: Callable[[str], GuardAction] | None = None,
)
```

`input_fn` and `output_fn` can be sync or async callables. `ContentGuard` detects awaitable returns automatically.

```python
import re

def redact_emails(text: str) -> GuardAction:
    cleaned = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', '[EMAIL]', text)
    if cleaned != text:
        return GuardAction.modify(cleaned, message="Email addresses redacted")
    return GuardAction.allow()

def require_disclaimer(text: str) -> GuardAction:
    if len(text) > 200 and "not financial advice" not in text.lower():
        return GuardAction.block("Missing required disclaimer")
    return GuardAction.allow()

# Input only
guard = ContentGuard(input_fn=redact_emails)

# Output only
guard = ContentGuard(output_fn=require_disclaimer)

# Both
guard = ContentGuard(input_fn=redact_emails, output_fn=require_disclaimer)

# Async input guard
async def async_pii_check(text: str) -> GuardAction:
    result = await external_api.scan(text)
    return GuardAction.block("PII detected") if result.has_pii else GuardAction.allow()

guard = ContentGuard(input_fn=async_pii_check)  # achat/aloop use it automatically
```

---

## 6. GuardChain — compose multiple guards

```python
GuardChain(guards: list)
```

Runs guards in sequence. **First block wins.** Modifications from earlier guards are passed to later guards (chains can stack redactions).

```python
length_guard    = ContentGuard(input_fn=lambda t: GuardAction.block("Too long") if len(t) > 5000 else GuardAction.allow())
pii_guard       = ContentGuard(input_fn=redact_emails)
disclaimer_guard = ContentGuard(output_fn=require_disclaimer)

guard = GuardChain([length_guard, pii_guard, disclaimer_guard])
resp = agent.chat("hello", guard=guard)
```

**Modification chaining:** if guard 1 modifies text, guard 2 sees the modified version. If any guard in the chain modifies and no guard blocks, `GuardChain` returns a single merged `GuardAction.modify(final_text)`.

---

## 7. LLMGuard — LLM as moderator

```python
LLMGuard(
    agent: LazyAgent,
    policy: str,
    check_input: bool = True,    # run on input?
    check_output: bool = False,  # run on output?
)
```

The moderator agent receives a structured prompt: content + policy. It must respond with `"ALLOW"` or `"BLOCK: <reason>"`. Any other response is treated as `ALLOW`. On exception, **fails closed** (blocks content) and logs a warning.

```python
from lazybridge import LazyAgent
from lazybridge import LLMGuard

moderator = LazyAgent("openai", model="cheap")  # use a cheap model; it's just classification

guard = LLMGuard(
    moderator,
    policy="Block any request that asks about weapons, illegal activity, or self-harm.",
    check_input=True,
    check_output=False,
)

resp = agent.chat("How do I make explosives?", guard=guard)
# raises GuardError("Blocked by LLM moderator")

# Input + output
guard = LLMGuard(moderator,
    policy="Inputs must be professional business questions. Outputs must not contain opinions.",
    check_input=True, check_output=True)
```

**Async:** `LLMGuard` implements `acheck_input` / `acheck_output` — uses `agent.atext()` internally. Safe to use with `achat` / `aloop`.

---

## 8. Where guards apply

Guards are accepted by: `chat()`, `achat()`, `loop()`, `aloop()`, `text()`, `atext()`, `json()`, `ajson()`.

```python
# Single call
resp = agent.chat("task", guard=guard)

# Loop (input guard on initial task; output guard on final response only)
resp = agent.loop("task", tools=[tool], guard=guard)

# Async
resp = await agent.achat("task", guard=guard)
```

**Loop behaviour:** the input guard runs once on the initial task string before the first LLM call. The output guard runs on `resp.content` after the loop completes (not on each intermediate step).

---

## 9. Implementing a custom Guard

```python
from lazybridge import Guard, GuardAction

class RateLimitGuard:
    """Block requests above N per minute."""
    def __init__(self, max_per_minute: int):
        import time, collections
        self._limit = max_per_minute
        self._times: collections.deque = collections.deque()

    def _check(self, text: str) -> GuardAction:
        import time
        now = time.monotonic()
        self._times = collections.deque(t for t in self._times if now - t < 60)
        if len(self._times) >= self._limit:
            return GuardAction.block(f"Rate limit: {self._limit} req/min exceeded")
        self._times.append(now)
        return GuardAction.allow()

    def check_input(self, text: str) -> GuardAction:
        return self._check(text)

    def check_output(self, text: str) -> GuardAction:
        return GuardAction.allow()

    async def acheck_input(self, text: str) -> GuardAction:
        return self._check(text)

    async def acheck_output(self, text: str) -> GuardAction:
        return GuardAction.allow()
```

---

## 10. Decision guide

| Use case | Guard type |
|---|---|
| Regex / rule-based filtering | `ContentGuard(input_fn=...)` |
| PII redaction | `ContentGuard(input_fn=redact_fn)` — use `modify()` not `block()` |
| Output policy (disclaimer, length) | `ContentGuard(output_fn=...)` |
| Multiple checks in sequence | `GuardChain([...])` |
| Semantic / intent moderation | `LLMGuard(moderator, policy=...)` |
| Custom stateful logic (rate limit, audit log) | Implement `Guard` protocol |

---

## 11. Production pattern

```python
from lazybridge import ContentGuard, GuardChain, LLMGuard, GuardAction, GuardError, LazyAgent
import re

# Layer 1: cheap rules (fast, no API call)
def block_long_input(text: str) -> GuardAction:
    return GuardAction.block("Input exceeds 4000 chars") if len(text) > 4000 else GuardAction.allow()

def redact_ssn(text: str) -> GuardAction:
    cleaned = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    return GuardAction.modify(cleaned) if cleaned != text else GuardAction.allow()

# Layer 2: LLM moderator (slower, semantic)
moderator = LazyAgent("openai", model="cheap")
llm_guard = LLMGuard(moderator, policy="Block requests about illegal activity.")

guard = GuardChain([
    ContentGuard(input_fn=block_long_input),
    ContentGuard(input_fn=redact_ssn),
    llm_guard,
])

try:
    resp = agent.chat(user_input, guard=guard)
except GuardError as e:
    # Log the block event; return safe error to user
    logger.warning("Request blocked: %s | metadata=%s", e.action.message, e.action.metadata)
    return {"error": "Request not allowed"}
```
