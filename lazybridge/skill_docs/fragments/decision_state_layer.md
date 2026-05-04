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
`Memory` is per-agent conversation history with compression. `Store` is
a shared, addressable blackboard (use `db=` for persistence). `sources=`
injects any live text into the system prompt at call time. All three
compose freely on the same agent.
