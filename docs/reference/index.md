# API reference

Auto-generated from the live docstrings in `lazybridge.*`. Pair with
the [Guides](../guides/basic/agent.md) for narrative explanations
and the [Recipes](../recipes/index.md) for runnable examples.

## Primary

- [Agent + Envelope](agent.md) — the universal wrapper and the
  typed result object every run produces.

## Tools

- [Tool family](tools.md) — `Tool` + the `Tool.wrap()` classmethod
  factory, the `ToolProvider` protocol, `NativeTool` enum.

## State

- [State primitives](state.md) — `Memory`, `Store`.
- [Session & observability](session.md) — `Session`, `EventLog`,
  `EventType`, `GraphSchema`, exporters.
- [Guards](guards.md) — `Guard`, `ContentGuard`, `GuardChain`,
  `LLMGuard`, `GuardError`.

## Engines & orchestration

- [Engines](engines.md) — `LLMEngine`, `Plan`, `Step`,
  `PlanCompileError`, `ToolTimeoutError`, `StreamStallError`.
- [Multi-agent graphs](multi-agent.md) — `AgentPool`, `conclude`,
  `ConcludeSignal` for dynamic, LLM-routed agent systems.
- [Sentinels & predicates](sentinels.md) — `from_*` sentinels and
  the `when` DSL for routing.

## Extensions

- [Extension engines](extensions.md) — `HumanEngine`,
  `SupervisorEngine`, MCP integration, `EvalSuite`, `Visualizer`,
  `OTelExporter`.
- [Custom providers](providers.md) — `BaseProvider` + the
  `LLMEngine.register_provider_*` registry.

## Configuration & testing

- [Runtime configs & testing](configs.md) — `CacheConfig` (kept) and
  `MockAgent`.  The 0.7-era ``AgentRuntimeConfig`` /
  ``ResilienceConfig`` / ``ObservabilityConfig`` were deleted in 0.7.9;
  fleet config uses a flat-kwarg dict spread now.
