# Memory

**Use `Memory` for** conversation history that survives multiple `agent.run()` calls.
The default `strategy="auto"` keeps things bounded (turn-window + summary) without
asking you to tune anything; switch to `"summary"` and pass a cheap `Agent` as
`summarizer=` once you need higher-fidelity context preservation.

**Don't use `Memory` for** durable cross-process state — that's `Store`. Memory's
job is "what should the model see in the next prompt"; Store's job is "what
should survive a crash".

`summarizer_timeout=30s` is a safety net: a stuck summariser must never block
the agent's tool loop. If the deadline fires you get a one-shot warning and
the keyword-extraction fallback runs.

## Example

```python
from lazybridge import Agent, Memory

mem = Memory(strategy="auto", max_tokens=3000)
chat = Agent("claude-opus-4-7", memory=mem, name="chat")

chat("hi, I'm Marco")
chat("what's my name?")         # "Marco"
print(mem.text())               # current compressed view

# LLM-summarised memory with explicit fallback timeout.
summariser = Agent("claude-haiku-4-5-20251001",
                   system="Summarize conversations concisely.")
mem = Memory(strategy="summary", summarizer=summariser, summarizer_timeout=15.0)

# Share memory across two agents — the judge reads the live history.
judge = Agent("claude-opus-4-7", name="judge",
              sources=[mem],
              system="Grade the assistant's last reply on helpfulness 1-5.")
judge("grade the last turn")
```

## Pitfalls

- ``Memory(strategy="summary")`` without a ``summarizer=`` agent uses
  the keyword-extraction fallback — bounded, but lossy. Pass a cheap
  agent for production-quality summaries.
- ``memory.clear()`` wipes everything including the in-process summary;
  it does not persist across restarts. For durable memory use ``Store``.
- ``max_turns`` is a hard backstop, not the primary compression knob.
  When it fires you get a one-shot warning — that's the signal to
  switch from ``strategy="none"`` to ``"auto"``.
- Setting ``summarizer_timeout=None`` restores the legacy unbounded
  behaviour. Do this only if you're confident your summariser is fast
  and reliable.

!!! note "API reference"

    Memory(
        *,
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int | None = 4000,         # token budget that triggers compression
        max_turns: int | None = 1000,          # hard cap on retained turns (memory backstop)
        store: Store | None = None,            # reserved — durable persistence (1.1+)
        summarizer: Agent | Callable | None = None,
        summarizer_timeout: float | None = 30.0,  # deadline applied to async summarisers
    ) -> Memory

    memory.add(user: str, assistant: str, *, tokens: int = 0) -> None
    memory.messages() -> list[Message]
    memory.text() -> str           # current view as plain text (live read)
    memory.clear() -> None

    Usage: Agent("model", memory=Memory("auto"))
           Agent("model", sources=[mem])     # share live view across agents

!!! warning "Rules & invariants"

    - ``auto`` — sliding window plus summary of older turns once
      ``max_tokens`` is exceeded; default. Good for general chat.
    - ``sliding`` — compress by dropping oldest turns whenever > 10 turns
      are kept. Does NOT require ``max_tokens``; works with ``max_tokens=None``.
    - ``summary`` — compress whenever > 10 turns are kept.
      Uses ``summarizer=`` if provided; otherwise falls back
      to keyword extraction (a rough but loss-aware fallback — never a
      silent no-op).
    - ``none`` — never compress; ``max_turns`` is the only backstop.
    - Failed structured-output retries (internal ``_validate_and_retry``
      loops) pass ``memory=None`` so correction turns are never stored as
      real conversation history.
    - ``Memory`` is per-agent by default. Share across agents by passing
      the same instance to each ``memory=`` or via ``sources=[mem]``.
    - ``text()`` is live — every call re-materialises the current view.
      Do not snapshot and cache it.
    - ``summarizer_timeout`` only enforces a deadline when the summariser
      returns a coroutine / awaitable. Sync summarisers cannot be
      cancelled mid-call; on timeout the keyword fallback runs.
    - Compression runs OUTSIDE the internal lock — concurrent ``add()``
      calls keep progressing while a slow summariser is in flight.

## See also

- [Store](store.md) — durable key-value store for cross-process state.
- [Session](session.md) — observability of `agent.run()` events.
