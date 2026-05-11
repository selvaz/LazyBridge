# Memory

Conversation history that survives multiple `agent(task)` calls. Per-agent
by default, shareable across agents, automatically bounded by a
sliding-window or LLM-summary compression strategy.

## Signature

```python
from lazybridge import Memory

Memory(
    *,
    strategy="auto",               # "auto" | "sliding" | "summary" | "none"
    max_tokens=4000,               # token budget that triggers compression
    max_turns=1000,                # hard backstop on retained turns
    store=None,                    # reserved — durable persistence (1.1+)
    summarizer=None,               # Agent or callable used by strategy="summary"
    summarizer_timeout=30.0,       # deadline for async summarisers (None = unbounded)
)

# Methods
mem.add(user, assistant, *, tokens=0)   # append a turn
mem.messages()                          # list[Message] for the LLM
mem.text()                              # current view as a plain string (live)
mem.clear()                             # wipe everything in process
```

Pass to an `Agent` via `memory=mem` (private to that agent) or
`sources=[mem]` (live read-only view shared across agents).

### Strategies

| `strategy=` | What it does | When |
|---|---|---|
| `"auto"` | Sliding window + summary of older turns once `max_tokens` is exceeded | General chat, the safe default |
| `"sliding"` | Drops oldest turns whenever > 10 are retained; works without `max_tokens` | Cheap, lossy, no LLM cost |
| `"summary"` | Compresses whenever > 10 turns are retained, using `summarizer=` (or a keyword-extraction fallback) | Higher fidelity at the cost of summariser tokens |
| `"none"` | Never compress; only `max_turns` bounds the buffer | You want full history and have explicit control over size |

## Synopsis

`Memory` is "what the model should see in the next prompt". It carries
conversation continuity across calls to the same agent (or across
several agents that share the instance). The default `"auto"` strategy
keeps the memory bounded without any tuning — sliding window first,
LLM summary of the older turns once the token budget is exceeded.

`Memory` is **not** durable. The whole buffer lives in the agent's
process and disappears when the process exits. For state that must
survive a crash or be shared across processes, use [Store](store.md).

## When to use it

- **Multi-turn conversations** with a single agent. Without `Memory`,
  every `agent(task)` call starts fresh; with it, the model sees the
  recent history.
- **Cross-agent shared context** when you want a judge or monitor
  agent to read the live conversation without writing to it. Pass the
  same `Memory` to the chat agent's `memory=` and the judge agent's
  `sources=[mem]`.
- **Bounded buffers in long-running interactive sessions** — the
  default `"auto"` strategy keeps token usage from growing without
  bound, and you don't have to do the trimming yourself.

## When NOT to use it

- **Durable cross-run state.** `Memory` doesn't survive process exit
  and isn't shared across machines. Use `Store` instead.
- **Pipeline data passing.** A `Plan` step's output flows to the next
  step via the envelope and sentinels (`from_prev`, `from_step("…")`),
  not via memory. Memory is for conversational context, not workflow
  state.
- **Structured-output retry loops.** When LazyBridge re-prompts the
  agent to fix an invalid structured-output payload, those correction
  turns are *not* added to memory — and neither should you add them
  manually.

## Example

```python
from lazybridge import Agent, LLMEngine, Memory


# 1) Default "auto" strategy — sliding window + summary fallback.
chat_memory = Memory(
    strategy="auto",
    max_tokens=3000,
)
chat = Agent(
    engine=LLMEngine("gemini-3-flash-preview"),
    memory=chat_memory,
    name="chat",
)

chat("hi, I'm Marco")
result = chat("what's my name?")
print(result.text())                  # "Marco"

print(chat_memory.text())             # current compressed view


# 2) "summary" strategy with a cheap summariser.
summariser = Agent(
    engine=LLMEngine(
        "claude-haiku-4-5-20251001",
        system="Summarize conversations concisely.",
    ),
)
high_fidelity_memory = Memory(
    strategy="summary",
    summarizer=summariser,
    summarizer_timeout=15.0,
)


# 3) Sharing live conversation across agents — chat writes, judge reads.
chat = Agent(
    engine=LLMEngine("gemini-3-flash-preview"),
    memory=chat_memory,
    name="chat",
)
judge = Agent(
    engine=LLMEngine(
        "claude-opus-4-7",
        system="Grade the assistant's last reply on helpfulness 1-5.",
    ),
    sources=[chat_memory],            # read-only live view
    name="judge",
)

chat("explain LazyBridge in one sentence")
print(judge("grade the last turn").text())
```

## Pitfalls

- **`strategy="summary"` without a `summarizer=`** falls back to
  keyword extraction — bounded, but lossy. Pass a cheap agent for
  production-quality summaries.
- **`memory.clear()`** wipes everything *including* the in-process
  summary; it does not persist across restarts. For durable memory
  use `Store`.
- **`max_turns`** is a hard backstop, not the primary compression
  knob. When it fires you get a one-shot warning — that's the signal
  to switch from `strategy="none"` to `"auto"`.
- **`summarizer_timeout=None`** restores the legacy unbounded
  behaviour. Use it only if you trust your summariser to be fast and
  reliable; otherwise a stuck summariser will block every `add(...)`
  call indefinitely.
- **`memory.text()` is a live read** — every call re-materialises the
  current view. Don't snapshot and cache it; if you need a stable
  reference for diagnostics, copy the string.
- **Sync summarisers can't be cancelled mid-call.** Only async
  summarisers (`async def` or returning a coroutine) honour
  `summarizer_timeout`. On timeout the keyword fallback runs.
- **Compression happens outside the internal lock**, so concurrent
  `add()` calls keep progressing while a slow summariser is in
  flight. This means a memory snapshot taken *during* compression may
  reflect the pre-compression view; that's intentional, but worth
  knowing if you're debugging.

## See also

- [Store](store.md) — the durable counterpart for cross-process
  state.
- [Session](session.md) — observability of `agent(task)` events;
  separate from memory.
- [Agent](../basic/agent.md) — the consumer (`memory=` for write
  access, `sources=[mem]` for live read).
