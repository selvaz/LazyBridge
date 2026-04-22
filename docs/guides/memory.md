# Memory

`Memory` is where an agent remembers the past. It records
user/assistant turns as they happen and exposes a compressed view the
next engine call can use as context. You never construct messages by
hand; just call `agent.run(task)` and pass `memory=Memory()` once.
All Memory constructor arguments are keyword-only
(`Memory(strategy="auto", max_tokens=3000)`, not positional).

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

## Example

```python
from lazybridge import Agent, LLMEngine, Memory

# strategy="auto"  → compress only once we blow past max_tokens.
# max_tokens=3000  → token budget; when the total estimated tokens in
#                    stored turns exceeds this, the oldest ones are
#                    compressed into a summary prefix.
mem = Memory(strategy="auto", max_tokens=3000)

# name="chat"      → label this agent carries into Session.graph,
#                    event logs, and usage_summary()["by_agent"].
#                    No functional effect on a single call; matters
#                    as soon as you add a Session or Plan.
chat = Agent("claude-opus-4-7", memory=mem, name="chat")

chat("hi, I'm Marco")
chat("what's my name?")         # "Marco"
print(mem.text())               # current compressed view (live, re-read each call)

# Share memory across two agents — the judge reads the live history
# via ``sources=[mem]`` (live view on every call). ``system=`` belongs
# on the engine, not the Agent.
judge = Agent(
    engine=LLMEngine("claude-opus-4-7",
                     system="Grade the assistant's last reply on helpfulness 1-5."),
    name="judge",             # distinct label in observability output
    sources=[mem],            # inject mem.text() into the system prompt each call
)
judge("grade the last turn")
```

## Persistence via `store=`

`Memory` is in-process by default.  If you want conversation state to
survive restarts, hand it a `Store` — each `add()` is mirrored to the
store and the Memory rehydrates from it at construction time.

```python
# What this shows: a chat agent whose memory survives process restarts.
# Why: a multi-turn assistant in a stateless web handler has to
# reconstruct the conversation every request. Persisting to SQLite is
# one line; everything else (compression, summarizer, live view)
# behaves exactly the same.

from lazybridge import Agent, Memory, Store

store = Store(db="chats.sqlite")
mem   = Memory(strategy="auto", max_tokens=4000, store=store)
chat  = Agent("claude-opus-4-7", memory=mem, name="chat")

chat("hi, my name is Marco")
# ... process restarts ...
# Next request reconstructs `mem` from `store` — the name is still there.
```

## LLM summarizer (async-safe)

For long conversations, keyword-extraction summaries are lossy.  A
cheap agent as `summarizer=` gives you an LLM-quality recap for a
fraction of the main agent's cost.  LazyBridge drives sync, `Agent`,
and plain `async def` summarizers transparently — you don't have to
match the surrounding event loop.

```python
# What this shows: using Haiku as a cheap summarizer for an Opus chat
# agent.  When compression triggers, the transcript of dropped turns
# is handed to Haiku and the returned summary becomes the "Context from
# earlier" prefix on subsequent calls.
# Why: Opus context windows are expensive; a three-cents-per-run
# summary tier keeps quality intact without paying Opus rates to
# re-read stale turns.

from lazybridge import Agent, LLMEngine, Memory

# A cheap, scoped summarizer — system prompt keeps it focused on
# preserving facts, not prose.
summarizer = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5",
        system="Summarise conversations tightly. Preserve names, "
               "decisions, and open questions. 2-4 sentences.",
    ),
    name="summarizer",
)

mem  = Memory(strategy="summary", summarizer=summarizer, max_tokens=6000)
chat = Agent("claude-opus-4-7", memory=mem, name="chat")
```

An `async def my_summariser(prompt: str) -> str: ...` coroutine works
identically — the Memory bridges the loop for you (via a worker
thread when called from inside a running loop) so the result is never
silently stringified as `<coroutine object …>`.

## Pitfalls

- Pass the same ``Memory`` instance to both agents if you want shared
  state. Copying or pickling resets the internal compression state.
- ``Memory(strategy="summary")`` without a ``summarizer=`` agent falls
  back to a keyword-extraction summary (``_rule_summary``) — lossy but
  non-empty. A hard ``max_turns=1000`` cap applies regardless of
  strategy so memory cannot grow unboundedly (pass ``max_turns=None``
  to opt out).
- ``memory.clear()`` wipes everything including the in-process summary;
  it does not persist across restarts. For durable memory use ``Store``.

!!! note "API reference"

    Memory(
        *,
        strategy: Literal["auto", "sliding", "summary", "none"] = "auto",
        max_tokens: int | None = 4000,
        max_turns: int | None = 1000,     # hard backstop; None disables
        store: Any | None = None,         # optional Store for persistence
        summarizer: Any | None = None,    # Agent or async callable used when strategy="summary"
    ) -> Memory
    
    memory.add(user: str, assistant: str, *, tokens: int = 0) -> None
    memory.messages() -> list[Message]
    memory.text() -> str              # current view as plain text (live read)
    memory.clear() -> None
    
    Usage: Agent("model", memory=Memory(strategy="auto"))

!!! warning "Rules & invariants"

    - ``auto`` — sliding window plus summary of older turns once ``max_tokens``
      is exceeded; default. Good for general chat.
    - ``sliding`` — always compresses once >10 turns accumulate, keeping
      only the last 10 verbatim. Lossy but cheap.
    - ``summary`` — same 10-turn cut-off as ``sliding``, but the compressed
      prefix is an LLM summary from ``summarizer=``; without a summarizer
      the fallback is keyword extraction (``_rule_summary``).
    - ``none`` — no token-based compression; only the ``max_turns`` hard cap
      applies.
    - ``Memory`` is per-agent by default. To share memory across agents, pass
      the same instance to each agent's ``memory=`` or via ``sources=[mem]``.
    - ``text()`` is live — every call re-materialises the current view. Do
      not snapshot and cache it.

## See also

[store](store.md), [agent](agent.md),
decision tree: [state_layer](../decisions/state-layer.md)
