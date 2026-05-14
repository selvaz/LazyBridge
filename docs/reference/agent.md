# Agent + Envelope

The universal wrapper and the typed result object every run produces.
For narrative usage see [Guides → Basic → Agent](../guides/basic/agent.md)
and [Guides → Basic → Envelope](../guides/basic/envelope.md).  If
you're migrating from a 0.7-era surface (`Agent.from_*` factories,
config dataclasses, `_ParallelAgent`), see the
[0.7 → 0.7.9 migration guide](../migrations/0.7-to-0.79.md).

::: lazybridge.Agent

::: lazybridge.ParallelAgent

::: lazybridge.Envelope

## Multimodal content blocks

For mixed-modality inputs (text + image + audio), pass `images=` and
`audio=` kwargs on `agent(...)`, `await agent.run(...)`, or
`async for chunk in agent.stream(...)`.  Bare URL strings, `Path`
objects, raw `bytes`, and `dict` payloads are coerced into the typed
blocks below automatically — use these constructors directly only when
you need to override the auto-detected MIME type.  Narrative coverage
lives in [Guides → Mid → Multimodal](../guides/mid/multimodal.md).

::: lazybridge.ImageContent

::: lazybridge.AudioContent
