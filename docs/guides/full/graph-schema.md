# GraphSchema

The agent topology view that a `Session` auto-populates as agents
register. Use it to inspect what your pipeline actually looks like
after construction, to render in an external diagramming tool, or
to persist as part of a run report.

`GraphSchema` is the **descriptor** layer — separate from the event
stream that [exporters](exporters.md) consume.

## Signature

```python
from lazybridge import GraphSchema, Session
from lazybridge.graph import NodeType, EdgeType

# Construction (rare — Session creates one for you).
GraphSchema(session_id="")


# Methods (auto-populated by Session; explicit calls are unusual).
graph.add_agent(agent)
graph.add_router(router)
graph.add_edge(from_id, to_id, *, label="", kind=EdgeType.TOOL)


# Inspection.
graph.nodes()                        # list[_BaseNode]
graph.edges()                        # list[Edge]
graph.edges_from(node_id)            # outgoing edges
graph.edges_to(node_id)              # incoming edges


# Serialisation.
graph.to_dict()
graph.to_json(indent=2)
graph.to_yaml()                      # requires lazybridge[yaml]
graph.save("topology.json")          # extension chooses format
GraphSchema.from_dict(d)
GraphSchema.from_json(s)
GraphSchema.from_file("topology.yaml")


# Enums
class NodeType:
    AGENT
    ROUTER

class EdgeType:
    TOOL                             # outer Agent uses inner via tools=[...]
    CONTEXT                          # data dependency (e.g. shared Memory)
    ROUTER                           # routing edge from a router node
```

## Synopsis

A `GraphSchema` records two kinds of nodes and three kinds of edges:

- **`AgentNode`** — every `Agent` constructed with `session=sess`
  gets one. Carries the agent's name, provider, model, and system
  prompt (read duck-typed from `agent.engine`).
- **`RouterNode`** — explicit router primitive (rare; most routing
  is encoded directly on `Step(routes=...)` and shows up as an
  edge instead).
- **`Edge(from_id, to_id, label, kind)`** — three flavours:
    - `TOOL` — outer agent's `tools=[...]` includes inner agent.
      `as_tool` wrappings are recorded automatically.
    - `CONTEXT` — data dependency (e.g. a shared `Memory`
      referenced by another agent's `sources=`).
    - `ROUTER` — routing edges from explicit router nodes.

The graph is **descriptor-only**. Reconstructing a runnable
pipeline from a saved graph is the caller's job — `from_dict` /
`from_json` give you the topology; you wire the actual `LLMEngine`,
`Agent`, `Plan` instances yourself.

## When to use it

- **Debugging.** `print(session.graph.to_json())` after a run
  confirms the pipeline you built matches what executed.
  Especially useful when you suspect a typo or missing
  `as_tool` registration.
- **External rendering.** Save the graph and load it in a
  diagramming tool, your own UI renderer, or a GraphViz pipeline.
  YAML / JSON output keeps the format human-friendly.
- **Run reports.** Persist the graph alongside event logs to
  document what the run looked like — useful for audits and
  post-mortems.
- **Pipeline diffs.** Hash or normalise the graph to detect
  unintentional changes to topology between deployments.

## When NOT to use it

- **Live event streams.** Use [exporters](exporters.md) instead;
  the graph captures *structure*, not *behaviour*.
- **Reconstructing a live pipeline from JSON.** `from_*` methods
  return descriptors only — they don't instantiate `Agent` /
  `LLMEngine` / `Plan` objects. For round-trippable Plan
  serialisation see *Plan.to_dict* (Phase 3c — Advanced).
- **Cost / token / latency aggregation.** That's
  `session.usage_summary()`, not the graph.

## Example

```python
from lazybridge import Agent, LLMEngine, Session


sess = Session()


researcher = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="researcher",
    session=sess,
)
writer = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="writer",
    session=sess,
)


# 1) The orchestrator's tools=[...] auto-registers as_tool edges.
orchestrator = Agent(
    engine=LLMEngine("claude-opus-4-7"),
    name="orchestrator",
    tools=[researcher, writer],
    session=sess,
)


# 2) Inspect what was built.
import json
print(json.dumps(sess.graph.to_dict(), indent=2))
# {
#   "session_id": "...",
#   "nodes": [
#     {"id": "researcher",   "type": "agent", ...},
#     {"id": "writer",       "type": "agent", ...},
#     {"id": "orchestrator", "type": "agent", ...},
#   ],
#   "edges": [
#     {"from": "orchestrator", "to": "researcher", "label": "as_tool", "kind": "tool"},
#     {"from": "orchestrator", "to": "writer",     "label": "as_tool", "kind": "tool"},
#   ]
# }


# 3) Persist + reload (descriptors only; not a runnable pipeline).
sess.graph.save("topology.yaml")

from lazybridge import GraphSchema
replay = GraphSchema.from_file("topology.yaml")
assert len(replay.nodes()) == 3
assert any(e.kind == "tool" for e in replay.edges())


# 4) Filter outgoing edges from the orchestrator.
for edge in sess.graph.edges_from("orchestrator"):
    print(f"{edge.from_id} -> {edge.to_id} ({edge.label})")


# 5) Render with your tool of choice — descriptor-only output is
#    tool-friendly. (graphviz is illustrative, not a dependency.)
import graphviz
g = graphviz.Digraph()
for node in sess.graph.nodes():
    g.node(node.id, label=node.id)
for edge in sess.graph.edges():
    g.edge(edge.from_id, edge.to_id, label=edge.label)
g.render("pipeline.gv")
```

## Pitfalls

- **An agent without `session=` is not registered anywhere.** If
  you pass it as a nested tool to an agent with a session, the
  outer agent propagates its session down and registers the
  nested one for you — but a top-level agent with no session
  produces no graph entry.
- **Duck-typed reads of `agent.engine.provider` and
  `agent.engine.model`.** A custom engine that doesn't expose
  those attributes leaves the corresponding `AgentNode` fields
  empty (`None`). If you need a custom name in the topology,
  set them explicitly on your engine.
- **`to_yaml` requires PyYAML.** Install via
  `pip install lazybridge[yaml]`. `to_json` is stdlib-only.
- **`from_dict` / `from_json` reconstruct descriptors only.** The
  `provider` / `model` strings on `AgentNode` are inert metadata;
  there's no live `LLMEngine` behind them. Don't try to
  `agent.run(...)` against a reloaded graph — wire your own
  agents with the same names if you want to replay.
- **The graph captures registration time, not runtime.** A
  conditional route that *could* fire shows up as an edge whether
  or not it actually fired in this run. For per-run "what
  actually happened", read the event log via
  `session.events.query(...)` instead.
- **Manual `add_edge` is rare.** Most edges register
  automatically when an agent is wrapped via `as_tool` (or passed
  directly into another agent's `tools=[...]`). Use
  `session.register_tool_edge(outer, inner, label=…)` only when
  wiring outside of `as_tool` (custom routing primitives, etc.).

## See also

- [Session](../mid/session.md) — owns the graph; populates it as
  agents register.
- [Exporters](exporters.md) — the live event stream; complements
  the graph's static topology view.
- [As tool](../mid/as-tool.md) — every `as_tool` wrapping records
  one `EdgeType.TOOL` edge automatically.
- *Guides → Advanced → Plan.to_dict* (Phase 3c) — round-trippable
  Plan serialisation, the runnable counterpart to graph
  descriptors.
