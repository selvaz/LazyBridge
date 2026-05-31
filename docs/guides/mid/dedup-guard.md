# DeduplicateGuard

Removes repeated text blocks from the agent's input **before** they reach
the LLM. Drop-in solution for the "recursive copy-paste" problem that
inflates context windows in multi-agent dialogue chains.

## The problem it solves

In a multi-agent pipeline each agent typically prepends the full conversation
history to the next agent's task.  After a few hops the same dialogue turn
can appear two, three, or ten times in the same context window:

```
[Turn 1] User: Summarise the report.
[Turn 2] Assistant: The report says…
[Turn 1] User: Summarise the report.   ← duplicate
[Turn 2] Assistant: The report says…   ← duplicate
[Turn 3] Supervisor: Refine the summary.
```

Every duplicated block wastes tokens, degrades coherence, and can push useful
content out of the context window.  `DeduplicateGuard` detects and removes
these blocks automatically — no prompt engineering needed.

## Quick start

```python
from lazybridge import Agent, DeduplicateGuard, LLMEngine

agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    guard=DeduplicateGuard(),
)
```

The guard fires only on `check_input`.  The LLM's output is never touched.

## Parameters

```python
DeduplicateGuard(
    similarity_chars: int = 60,   # prefix length for near-duplicate detection
    min_block_chars:  int = 40,   # blocks shorter than this are never removed
    verbose:          bool = True, # print a one-line summary when blocks removed
)
```

| Parameter | Default | What it controls |
|---|---|---|
| `similarity_chars` | `60` | How many leading chars are used as a "fingerprint". Lower = more aggressive. |
| `min_block_chars` | `40` | Minimum block length to be eligible for dedup. Prevents removing short repeated phrases like "Yes" or "Ok". |
| `verbose` | `True` | Prints `[DeduplicateGuard] removed N block(s) — X → Y chars (-Z%)` to stdout. Set `False` in production. |

## How it works

### 1 — Block splitting

The guard splits the input into *blocks* using the first strategy that
produces more than one block:

1. **`[Turn N]` markers** — dialogue turns in multi-agent history
2. **Double newlines** — paragraphs
3. **Single newlines** — individual lines

### 2 — Normalisation

Each block is normalised before comparison: whitespace collapsed, lowercased.
This means `"  Hello  World  "` and `"hello world"` are considered the same
block.

### 3 — Fingerprint matching

A block is dropped if either condition holds:

- Its **full normalised form** matches a previously seen block (exact duplicate).
- Its **first `similarity_chars` characters** match a previously seen block
  (near-duplicate — catches blocks that share an intro but diverge slightly).

### 4 — Re-joining

The kept blocks are re-joined with the same separator as the original input
(`\n`, `\n\n`, or no separator) so the cleaned text is a drop-in replacement.

## Tuning

### Dialogue chains (default settings work well)

```python
# Default: aggressively deduplicate [Turn N] markers
DeduplicateGuard()
```

### Long structured documents

If your context contains long sections (e.g., code blocks, JSON payloads)
that legitimately start with the same prefix, raise `similarity_chars` to
avoid false positives:

```python
DeduplicateGuard(similarity_chars=120)
```

### Short repeated phrases are meaningful

If your domain has many short repeated commands that should be preserved,
raise `min_block_chars`:

```python
DeduplicateGuard(min_block_chars=80)
```

### Silent operation in production

```python
DeduplicateGuard(verbose=False)
```

## Combining with other guards

`DeduplicateGuard` slots into a `GuardChain` like any other guard.
Run it **first** so downstream guards operate on the already-cleaned text:

```python
import re
from lazybridge import Agent, ContentGuard, DeduplicateGuard, GuardAction, GuardChain, LLMEngine

def no_pii(text: str) -> GuardAction:
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return GuardAction(allowed=False, message="Remove email addresses.")
    return GuardAction(allowed=True)

agent = Agent(
    engine=LLMEngine("claude-haiku-4-5"),
    guard=GuardChain(
        DeduplicateGuard(verbose=False),   # 1. clean duplicates first
        ContentGuard(input_fn=no_pii),     # 2. then check policy
    ),
)
```

## Example: supervisor chain

```python
from lazybridge import Agent, DeduplicateGuard, LLMEngine

researcher = Agent(engine=LLMEngine("claude-haiku-4-5"), name="researcher")
supervisor = Agent(
    engine=LLMEngine("claude-sonnet-4-6"),
    guard=DeduplicateGuard(),   # protect the supervisor from bloated history
    name="supervisor",
)

# Build a history string and pass it to the supervisor.
history = "\n".join([
    "[Turn 1] User: Summarise Q3 sales.",
    "[Turn 2] Researcher: Q3 revenue was $4.2M…",
    "[Turn 1] User: Summarise Q3 sales.",   # duplicated by the pipeline
    "[Turn 2] Researcher: Q3 revenue was $4.2M…",  # duplicated
])

result = supervisor(f"{history}\n[Turn 3] Supervisor: Refine this summary.")
# The guard removed 2 duplicate blocks before the LLM saw the input.
```

## The `deduplicate()` function

The underlying function is also available standalone if you want to clean
text without attaching a guard to an agent:

```python
from lazybridge.dedup_guard import deduplicate

cleaned, n_removed = deduplicate(
    "[Turn 1] Hello\n[Turn 1] Hello\n[Turn 2] World",
    similarity_chars=60,
)
print(cleaned)    # "[Turn 1] Hello\n[Turn 2] World"
print(n_removed)  # 1
```

## Pitfalls

- **Not for semantic deduplication.** The guard uses exact and prefix
  matching, not embedding similarity.  Two blocks that say the same thing
  in different words will both pass through.  For semantic dedup, combine
  with an `LLMGuard` that detects content overlap.

- **Block splitting is heuristic.** If your input mixes `[Turn N]` markers
  and double-newline paragraphs in the same string, the guard uses
  `[Turn N]` markers and ignores paragraph breaks within a turn.

- **`min_block_chars` applies to the original block, not the normalised
  form.** Whitespace-heavy blocks that are short after normalisation may
  still pass the length check.

- **The guard modifies the task string, not the message list.** If you are
  passing a structured `messages=` list directly to the engine rather than
  a plain string task, the guard does not inspect individual messages.
  Use it at the outermost agent boundary where the full history is
  concatenated into a single string.

- **`verbose=True` writes to stdout.** In production or in test suites
  that assert on stdout, set `verbose=False`.

## See also

- [Guards](guards.md) — guard protocol, `ContentGuard`, `LLMGuard`,
  `GuardChain`, and when to use hard gates vs soft verify.
- [Session](session.md) — `DeduplicateGuard` emits no session events
  (it rewrites silently); add your own `ContentGuard` with `metadata=`
  if you need observability of dedup decisions.
- [Reference → Guards](../../reference/guards.md) — full API surface.
