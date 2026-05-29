# Multi-agent graphs

Primitives for dynamic, LLM-routed multi-agent systems where the topology
is decided at runtime rather than declared up front.

- `AgentPool` — a name-keyed registry exposed to agents as a single
  `route(agent_name, task)` tool. Lets agents delegate to **each other**
  by name (resolving the circular-reference problem that a frozen tool map
  otherwise imposes) and bounds recursion via `max_depth`.
- `conclude` — a non-local exit: any agent, however deeply nested, can call
  `conclude("answer")` to end the whole task and return its answer straight
  to the originating top-level `Agent.run`. Raised internally as
  `ConcludeSignal` (a `BaseException`) and caught only at the top level.

Pair them with `LLMEngine(max_tool_calls_per_turn=1)` to keep the graph on a
single non-branching path. For narrative usage see
[Guides → Mid → Dynamic graph](../guides/mid/dynamic-graph.md).

::: lazybridge.AgentPool

::: lazybridge.conclude

::: lazybridge.ConcludeSignal
