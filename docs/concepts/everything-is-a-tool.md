# Everything is a tool

> If something exposes a useful capability, it should be usable as a tool —
> regardless of whether it is a function, an agent, a plan, an MCP server, or
> a whole pipeline.

This is the composition rule that holds the framework together. Most agent
frameworks distinguish sharply between functions, agents, chains,
workflows, plugins, integrations, and tools. LazyBridge collapses these
into a single concept.

## What can be a tool

| Source | How | Where it lives |
|---|---|---|
| A Python function | Pass it directly to `Agent(tools=[...])` | Your code |
| Any callable | Wrap with `Tool(...)` | Your code |
| Another `Agent` | `agent.as_tool(name, description)` | Hierarchical / supervisor patterns |
| A `Plan` | Wrap an `Agent.from_plan(...)` agent and call `.as_tool(...)` | Reusable deterministic pipelines |
| A provider-native capability | `NativeTool.WEB_SEARCH`, `NativeTool.CODE_EXECUTION`, … | Provider-side, no code |
| An MCP server | `MCPServer.stdio(...)` or `MCPServer.http(...)` | External tool ecosystems |
| A pre-built JSON schema | `Tool.from_schema(name, description, parameters, func)` | OpenAPI bridges, third-party registries |

In every case, the agent that consumes the tool sees the same `Tool`
object. There is no special-case glue for "agent calls function" versus
"agent calls another agent" versus "agent calls MCP server".

## Composition is recursive

Because every capability is a tool, you can compose at every level.

```text
   plain function
        │
        ▼  Tool(...)
       tool ──────────► added to an Agent
                            │
                            ▼  agent.as_tool("search", ...)
                          tool ──────────► added to a higher-level Agent
                                                │
                                                ▼  Agent.from_plan(...).as_tool(...)
                                              tool ──────────► added to a top-level orchestrator
```

A two-line illustration:

```python
researcher = Agent.from_model("claude-sonnet-4-6", tools=[web_search, fetch_url])
writer     = Agent.from_model("claude-sonnet-4-6", tools=[researcher.as_tool("research", "Look things up online")])
```

The `writer` agent now has a tool called `research` whose implementation
is a fully-fledged sub-agent with its own model, prompt, tools, and cost
tracking. The supervisor pattern, hierarchical planning, and "agents as
tools" are all the same primitive.

## Why this matters

**One thing to learn.** You don't need separate concepts for "function
calling", "tool use", "sub-agents", "delegation", "subgraphs", or "node
edges". You only need to learn how a `Tool` works.

**One contract to test.** Every tool — function, agent, MCP, plan — speaks
the same input/output contract. A `MockAgent` is a drop-in replacement for
a real one. Your test fixtures stay readable.

**Free recursion.** Cost, token counts, latency, and errors roll up
transitively through nested tool calls. A `daily_news_report` agent that
calls a `region_pipeline` that calls a `writer_agent` reports total cost
and total tokens at the top, regardless of how many levels are nested in
between.

**No orchestration glue.** A pipeline is just a `Plan`. A `Plan` is a tool.
A tool is something an agent already knows how to call. You don't write
glue — you compose.

## The trade-off worth naming

"Everything is a tool" tempts you to over-decompose. **Sub-agents are not
free.** Every nested agent adds latency, an LLM call (if its engine is an
LLM), and prompt-construction overhead.

Use a sub-agent when:

- It has a **distinct responsibility** with a clear input and output.
- A **smaller / cheaper / specialised model** is appropriate.
- The output type is **structured** and the parent agent benefits from a
  validated payload rather than free-form text.
- You want to **reuse** the same capability from multiple parents.

Don't use a sub-agent merely because the diagram looks nicer with one.
A single agent with three tools often beats three agents with one tool
each.

## See also

- [Mental model](mental-model.md) — where Tools sit in the
  Engine + Tools + State decomposition.
- [Progressive complexity](progressive-complexity.md) — when "wrap the
  agent as a tool" is the right next step versus when to reach for a
  `Plan`.
- *Guides → Tool* (coming in Phase 2) — the three schema modes
  (`signature` / `llm` / `hybrid`) and the `Tool.from_schema` escape
  hatch.
- *Guides → As tool* (coming in Phase 2) — the full mechanics of
  `agent.as_tool(...)` including the optional verifier loop.
- *Guides → MCP* (coming in Phase 2) — connecting external tool
  ecosystems through stdio and HTTP transports.
