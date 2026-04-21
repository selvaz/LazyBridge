## question
State: Memory, Store, or sources=?

## tree
Conversation history for one agent?
    → Memory         # agent.memory = Memory("auto")

Shared key-value blackboard across multiple agents / runs?
    → Store          # store.write / store.read

Static documents (files, URLs, strings) injected into context at call time?
    → sources=[...]  # callable, Memory, Store, or raw string

Multiple patterns at once?
    → Yes — they compose.
      Example: Agent(memory=Memory(), sources=[shared_store, policy_text])

## tree_mermaid
flowchart TD
    A[Where does my state live?] --> B{Scope?}
    B -->|per agent, per conversation| C[Memory]
    B -->|shared across agents| D[Store]
    B -->|static / external / live view| E[sources=]
    C & D & E -->|compose freely| F[Pass all three if needed]

## notes
`Memory` is conversational and per-agent. It records turns and
compresses older ones when your token budget is exceeded.

`Store` is a blackboard: explicit, addressable by key, shareable.
Use it when agents need to hand off intermediate state or cache results
across runs. Pass ``db="file.sqlite"`` for persistence.

`sources=[...]` is context injection. Each source object is asked for
its current text at call time (live view — no snapshotting), and the
concatenated text is appended to the system prompt. Sources can be
`Memory`, `Store`, callables, or plain strings.

All three compose: an agent can have its own `memory`, read from a
shared `Store` via `sources=`, and also inject a policy string.
