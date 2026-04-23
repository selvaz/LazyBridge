# State: Memory, Store, or sources=?

```mermaid
flowchart TD
    A[Where does my state live?] --> B{Scope?}
    B -->|per agent, per conversation| C[Memory]
    B -->|shared across agents| D[Store]
    B -->|static / external / live view| E[sources=]
    C & D & E -->|compose freely| F[Pass all three if needed]
```

`Memory` is per-agent conversation history with compression. `Store` is
a shared, addressable blackboard (use `db=` for persistence). `sources=`
injects any live text into the system prompt at call time. All three
compose freely on the same agent.
