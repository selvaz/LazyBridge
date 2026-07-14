# Tool wrapping

> **Turning a library function into a tool: `Tool.wrap(fn, ...)` directly, a
> shared-state handle, or a bridging class?**

Pick by **what's actually missing** — usually nothing is, and the bridging
class is boilerplate that re-declares a signature `Tool.wrap` already reads.

## Decision tree

```text
Does a plain function (or a small set of them) with type hints already do
the work, one call in, one JSON-safe result out?
    → Tool.wrap(fn, name="...")
      # zero bridging code — LazyBridge reads the signature/docstring
      # natively (mode="signature", the default)

Do several tools need the SAME loaded/expensive state (a big array, a
parsed document, a fitted model) without re-loading it on every call?
    → One tool loads it and returns a handle (a Store/depot key); every
      other tool takes that handle as a parameter and reads the state back
      by key.
      # Share state via a KEY passed through the LLM, never the state
      # itself — and never give each tool its own private loader for the
      # same underlying data.

Does the raw function's result need capping/redacting for LLM context
(an unbounded list, a huge array, a secret) or a specific envelope /
provenance shape a caller depends on?
    → A thin wrapper IS justified — but wrap ONLY the missing concern (the
      cap, the envelope) and call the library function unmodified inside
      it. The wrapper's parameters should still mirror the library
      function's, not invent a parallel shape.

Is the tool's schema already known ahead of time (an MCP tool catalogue,
an OpenAPI operation, a third-party registry) rather than introspectable
from a Python callable?
    → Tool.from_schema(name, description, parameters, func, ...)
      # the canonical no-signature path — not a bridging class either.

None of the above — you're re-declaring the function's own parameter
list, restating its docstring as a new description=, and/or reloading
data the function (or a sibling tool) could already read from a shared
handle?
    → You're writing boilerplate. Delete the wrapper; call Tool.wrap(fn,
      ...) on the library function directly.
```

## Quick reference

| Situation | Use |
|---|---|
| Plain function, self-contained | `Tool.wrap(fn, name=...)` |
| Several tools need one loaded resource | A shared store handle (a `*_key` parameter every tool reads by) |
| Output needs an LLM-context cap or an envelope/provenance shape | A thin wrapper — cap/envelope only, delegate the computation unchanged |
| Schema is already known (MCP/OpenAPI/registry) | `Tool.from_schema(...)` |
| Wrapper re-declares the function's own signature | Delete it; wrap directly |

## Notes

- **The signature is the schema.** `Tool.wrap(fn, name=...)` (`mode="signature"`,
  the default) introspects type hints and the docstring — including
  `Annotated[type, "description"]` per-parameter docs — into the tool's JSON
  schema. A hand-written bridge method that re-types the same parameters and
  re-writes the same description is strictly worse: two places can drift,
  and only one of them is what the library actually does.
- **Share expensive state by key, not by tool.** When two or more tools need
  the same loaded data (a returns matrix, a parsed corpus, a fitted model),
  the library function itself should accept a store-key parameter and read
  from a shared store — not each tool re-implementing its own loader that
  happens to fetch the same data. One loader tool, N consumer tools, one
  handle passed between them.
- **A cap is not a re-implementation.** Wrapping a function to bound its
  output for LLM context (e.g. "return the 250 most recent items, not all
  10,000") is legitimate bridge code — but the wrapper should still call the
  unmodified library function and only touch the part of the result that
  needs bounding, not rebuild the whole computation.
- **This applies one level up too, with a different verdict.** An agent
  that exists only to hold one `ToolProvider` plus a tailored system prompt
  (a "specialist") is *not* the boilerplate this page warns about — the
  system prompt is real content a raw tool list doesn't carry. The
  distinction is: a specialist agent adds a prompt around existing tools; a
  bridging class re-implements the tools themselves. Avoid the latter, not
  the former.

## See also

- [Tool](../guides/basic/tool.md) — schema modes (`signature` / `hybrid` /
  `llm`) and when each is appropriate.
- [Everything is a tool](../concepts/everything-is-a-tool.md) — the
  composition philosophy this decision sits inside.
- [LLM codegen contract](../for-llms/codegen-contract.md) — the terse
  Always/Never version of this rule.
- Worked example: [LazyTools' regime detection](https://github.com/selvaz/LazyTools/blob/main/docs/regimes.md)
  connector — every tool is a one-line `Tool.wrap` of a `lazystats`
  function; one tool loads data into a shared depot under a `data_key`
  handle, and every fitting tool reads that same handle instead of
  loading its own copy.
