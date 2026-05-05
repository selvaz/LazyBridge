# Basic tier

**Use this when** you need a single LLM call — with or without tools, with or without structured output. No setup beyond an API key.

**Move to Mid when** you need memory across calls, shared state, tracing, guardrails, or more than one agent in sequence.

## Walkthrough

Start at the top: an `Agent` is a thin wrapper that delegates to an
`Engine`.  The default engine is `LLMEngine`, so
`Agent("claude-opus-4-7")` is everything you need for a one-shot call.

**Tools** are the next concept.  Pass any Python function with type
hints and a docstring as `tools=[...]` and the framework builds the
JSON schema automatically — see [Function → Tool](../guides/tool-schema.md)
for when the signature path isn't enough.  **Native tools** are
provider-hosted alternatives (web search, code execution): pass an
enum, no code.

Every call returns an **`Envelope`** — a typed wrapper carrying
`payload` (the result), `metadata` (token / cost / latency), and
`error`.  Use `.text()` for "give me a string"; `.payload` for the
typed Pydantic model when `output=` is set; `.ok` to check error state.

## Topics

* [Agent](../guides/agent.md)
* [Tool](../guides/tool.md)
* [Native tools (web search, code execution, …)](../guides/native-tools.md)
* [Function → Tool (schema modes)](../guides/tool-schema.md)
* [Envelope](../guides/envelope.md)

## Next steps

* Recipes: [Tool calling](../recipes/tool-calling.md) · [Structured output](../recipes/structured-output.md)
* [Mid tier →](mid.md) when you need state, tracing, or composition

**By the end of Basic you can:**

- call any LLM with `Agent("model")(task)`
- turn any typed Python function into a tool with auto-schema
- opt into provider-hosted native tools (web search, code execution)
- get a typed Pydantic payload back instead of plain text
