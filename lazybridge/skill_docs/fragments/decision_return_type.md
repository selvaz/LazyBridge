## question
What does my agent return — text, typed object, or metadata?

## tree
Plain string response?
    → .text()                 # str, always, regardless of output type

Pydantic model / structured result?
    → Agent("model", output=MyModel)(task).payload   # MyModel instance

Need token count, cost, latency, run id?
    → env.metadata            # EnvelopeMetadata dataclass

Need to check for errors before reading payload?
    → if env.ok: ... else: env.error.message

## tree_mermaid
flowchart TD
    A[Agent returns Envelope] --> B{What do you need?}
    B -->|string| C[env.text]
    B -->|typed object| D[env.payload with output=Model]
    B -->|tokens / cost| E[env.metadata]
    B -->|error check| F[env.ok then env.error]

## notes
An ``Envelope`` carries everything the engine knows about a run: the
payload (string by default; typed when ``output=`` is set), metadata
(tokens, cost, latency, run id), and an optional error channel. You
pick what you read; nothing is hidden.

Calling ``.text()`` is safe on every Envelope — it serialises Pydantic
payloads as JSON and handles ``None`` as empty string.
