## signature
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

## rules
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

## narrative
`Memory` is where an agent remembers the past. It records
user/assistant turns as they happen and exposes a compressed view the
next engine call can use as context. You never construct messages by
hand; just call `agent.run(task)` and pass `memory=Memory()` once.

The four strategies trade off fidelity against token budget. `auto` is
the right default for chat-like interactions. `sliding` is the right
default when you just want the last N turns and don't care about older
context. `summary` matters when conversations get long and you need a
distilled recap instead of raw transcripts. `none` is for tests and
unit-length scripts — it grows unboundedly.

Because `memory.text()` re-reads on every call, you can also pass a
`Memory` to `Agent(..., sources=[memory])` on a *different* agent.
That agent will see the live conversation view without owning it —
useful for "shadow" observers that summarise or judge.

## example
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

## pitfalls
- Pass the same ``Memory`` instance to both agents if you want shared
  state. Copying or pickling resets the internal compression state.
- ``Memory(strategy="summary")`` without a ``summarizer=`` agent falls
  back to a no-op and grows unboundedly.
- ``memory.clear()`` wipes everything including the in-process summary;
  it does not persist across restarts. For durable memory use ``Store``.

## see-also
[store](store.md), [agent](agent.md),
decision tree: [state_layer](../decisions/state-layer.md)
