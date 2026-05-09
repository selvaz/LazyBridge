# Do I need Advanced?

> **Smell test for the framework-author tier.**

Advanced is for framework authorship, not application development.
If you're tweaking prompts, swapping models, or building pipelines,
you're in Basic / Mid / Full.

## Decision tree

```text
Adding support for a new LLM vendor (Mistral, Cohere, Bedrock,
in-house model, …)?
    → Yes, Advanced.   BaseProvider + LLMEngine.register_provider_rule

Replacing the execution loop with a non-LLM strategy (rules,
deterministic dispatch, RL, recorded-script driver, …)?
    → Yes, Advanced.   Engine protocol

Serialising a Plan to disk / over the wire / between services?
    → Yes, Advanced.   Plan.to_dict / Plan.from_dict

Wiring deeper OpenTelemetry — custom TracerProvider, in-memory
exporter for tests, custom resource attributes?
    → Yes, Advanced.   OTelExporter(exporter=…)

Building a custom UI on top of the live event stream?
    → Yes, Advanced.   Visualizer / EventHub / HubExporter

Importing from `lazybridge.core.*` directly in app code?
    → STOP. Smell test failed — you probably want the public surface
      from `lazybridge import …`.

Just tweaking prompts / swapping models / building pipelines?
    → STOP. Basic / Mid / Full cover this.
```

## Quick reference

| Are you writing… | Tier |
|---|---|
| A provider adapter | **Advanced** — `BaseProvider` |
| A new execution strategy | **Advanced** — `Engine` protocol |
| A plan serialiser / worker pool | **Advanced** — `Plan.to_dict` |
| A custom OTel pipeline | **Advanced** — `OTelExporter(exporter=…)` |
| A custom UI on session events | **Advanced** — `EventHub` / `HubExporter` |
| App code importing `lazybridge.core.types` | **Probably not** — reconsider |
| Pipelines / prompts / models | **Basic / Mid / Full** |

## Notes

- **Advanced is opt-in.** None of the Basic / Mid / Full surface
  references Advanced primitives — you can ship a pipeline to
  production without ever opening the Advanced docs.
- **Smell test: imports from `lazybridge.core.*` in app code.**
  `from lazybridge import Agent, Tool, Envelope, Memory, Store,
  Session, Plan, Step, …` covers 99% of real use. If you find
  yourself reaching for `lazybridge.core.types`, step back —
  the public `Envelope` / `Agent` / `Tool` usually has what
  you need with a friendlier surface.
- **Stable contracts.** The Advanced surface (`BaseProvider`,
  `Engine` protocol, `Plan.to_dict`) is stable across minor
  versions; breaking changes follow a deprecation cycle plus a
  minor-version bump. Depend on it confidently when you need
  it.
- **Phase upgrades carefully.** Adding a custom provider or
  engine is a code-review-worthy change because it bypasses the
  framework's tested integrations. Pair with regression tests
  against your custom path.

## See also

- [BaseProvider](../guides/advanced/base-provider.md) — the
  stable extension point for new LLM backends.
- [Engine protocol](../guides/advanced/engine-protocol.md) —
  how to write a non-LLM execution layer.
- [Plan serialization](../guides/advanced/plan-serialize.md) —
  `to_dict` / `from_dict` round-trip.
- [OpenTelemetry](../guides/advanced/otel.md) — deep dive on
  semantic conventions and custom tracer setup.
- [Visualizer](../guides/advanced/visualizer.md) — live and
  replay UI; custom UI hooks via `EventHub`.
