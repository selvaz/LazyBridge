# What does my agent return — text, typed object, or metadata?

```mermaid
flowchart TD
    A[Agent returns Envelope] --> B{What do you need?}
    B -->|string| C[env.text]
    B -->|typed object| D[env.payload with output=Model]
    B -->|tokens / cost| E[env.metadata]
    B -->|error check| F[env.ok then env.error]
```

An ``Envelope`` carries everything the engine knows about a run: the
payload (string by default; typed when ``output=`` is set), metadata
(tokens, cost, latency, run id), and an optional error channel. You
pick what you read; nothing is hidden.

Calling ``.text()`` is safe on every Envelope — it serialises Pydantic
payloads as JSON and handles ``None`` as empty string.
