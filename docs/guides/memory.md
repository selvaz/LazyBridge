# Memory

## Example

```python
from lazybridge import Agent, Memory

mem = Memory("auto", max_tokens=3000)
chat = Agent("claude-opus-4-7", memory=mem, name="chat")

chat("hi, I'm Marco")
chat("what's my name?")         # "Marco"
print(mem.text())               # current compressed view

# Share memory across two agents — the judge reads the live history.
judge = Agent("claude-opus-4-7", name="judge",
              sources=[mem],
              system="Grade the assistant's last reply on helpfulness 1-5.")
judge("grade the last turn")
```

## Pitfalls

- ``Memory(strategy="summary")`` without a ``summarizer=`` agent falls
  back to a no-op and grows unboundedly.
- ``memory.clear()`` wipes everything including the in-process summary;
  it does not persist across restarts. For durable memory use ``Store``.

!!! note "API reference"

    Memory(
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int = 4000,
        window: int = 10,            # sliding: last N turns kept raw
        summarizer: Agent | None = None,  # cheap agent used when strategy="summary"
    ) -> Memory
    
    memory.add(user: str, assistant: str, tokens: int = 0) -> None
    memory.messages() -> list[Message]
    memory.text() -> str              # current view as plain text (live read)
    memory.clear() -> None
    
    Usage: Agent("model", memory=Memory("auto"))

!!! warning "Rules & invariants"

    - ``auto`` — sliding window plus summary of older turns once ``max_tokens``
      is exceeded; default. Good for general chat.
    - ``sliding`` — keep last ``window`` turns verbatim, drop the rest.
      Lossy but cheap.
    - ``summary`` — compress everything with ``summarizer`` on each overflow.
      Requires passing a cheap Agent.
    - ``none`` — do not compress; raises memory over time.
    - ``Memory`` is per-agent by default. To share memory across agents, pass
      the same instance to each agent's ``memory=`` or via ``sources=[mem]``.
    - ``text()`` is live — every call re-materialises the current view. Do
      not snapshot and cache it.

