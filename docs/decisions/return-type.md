# Return type

> **What does my agent return — text, typed object, or metadata?**

Every `agent(task)` returns an `Envelope`. What you read off it
depends on what you asked for.

## Decision tree

```text
Plain string response?
    → result.text()                # str, always, regardless of payload type

Pydantic model / structured result?
    → Agent(engine=LLMEngine("…"), output=MyModel)
      ...
      result.payload               # MyModel instance

Token count, cost, latency, run id?
    → result.metadata              # EnvelopeMetadata

Need to check for errors before reading payload?
    → if result.ok:
          read result.payload
      else:
          read result.error.message
```

## Quick reference

| You want | Read |
|---|---|
| Stringified response (works for any payload type) | `result.text()` |
| Validated typed payload | `result.payload` (with `output=PydanticModel`) |
| Token / cost / latency / model / provider | `result.metadata` |
| Did this run succeed? | `result.ok` |
| Why did it fail? | `result.error.type` / `result.error.message` |
| Original task | `result.task` |

## Notes

- **`.text()` is always safe.** It serialises Pydantic payloads
  as JSON, returns `""` for `None`, and returns the string
  verbatim for `str` payloads. Use it whenever you want a string
  regardless of the payload shape.
- **`output=Model` + `.text()` returns JSON, not human-readable
  text.** With structured output, read `.payload` directly to
  get the model instance. Calling `.text()` on a structured
  envelope is a common confusion — see
  [Envelope](../guides/basic/envelope.md) pitfalls.
- **`metadata.nested_*` aren't authoritative for cross-agent
  rollup.** They reflect what flowed through *this* envelope's
  lineage, not the entire run. For an authoritative
  cross-agent cost, query `session.usage_summary()`.
- **Always check `.ok` before reading `.payload` in production.**
  An error envelope's payload is whatever was last produced (or
  `None`); it's `result.error` that tells you something went
  wrong.

## See also

- [Envelope](../guides/basic/envelope.md) — full reference for
  the `Envelope` shape, `EnvelopeMetadata`, `ErrorInfo`.
- [Agent](../guides/basic/agent.md) — `output=` kwarg behaviour
  and structured-output retry.
