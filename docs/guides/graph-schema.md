# GraphSchema

`GraphSchema` is the topology view of a `Session`. Every agent
constructed with `session=s` registers a node; every `Agent.as_tool()`
call from within that session records an `as_tool` edge. The result is
a live-updating graph you can serialise, inspect, or hand to a GUI.

Two typical uses:

* **Debug / docs** — `print(session.graph.to_json())` after a run to
  confirm the pipeline you think you built is the one that ran.
* **External rendering** — dump to JSON/YAML for a UI (`save("g.yaml")`
  → load in a diagramming tool, or feed to your own renderer).

`GraphSchema` round-trips through `to_dict` / `from_dict`, so you can
persist a topology, edit it, and reload — handy for saved workflow
templates.

## Example

```python
from lazybridge import Agent, Session

sess = Session()
researcher = Agent("claude-opus-4-7", name="researcher", session=sess)
writer     = Agent("claude-opus-4-7", name="writer",     session=sess)

orchestrator = Agent(
    "claude-opus-4-7",
    name="orchestrator",
    tools=[researcher, writer],   # as_tool edges registered automatically
    session=sess,
)

print(sess.graph.to_json(indent=2))
# {
#   "session_id": "...",
#   "nodes": [AgentNode(researcher), AgentNode(writer), AgentNode(orchestrator)],
#   "edges": [
#     Edge(from=orchestrator, to=researcher, label="as_tool", type="tool"),
#     Edge(from=orchestrator, to=writer,     label="as_tool", type="tool"),
#   ]
# }

# Persist + reload.
sess.graph.save("topology.yaml")
from lazybridge import GraphSchema
replay = GraphSchema.from_file("topology.yaml")
assert len(replay.nodes()) == 3
```

## The three edge kinds — `TOOL`, `CONTEXT`, `ROUTER`

An edge is more than a line: its `kind` tells the reader (and any
downstream renderer) what kind of relationship it represents.
LazyBridge auto-populates TOOL edges from `as_tool` wrapping; you add
CONTEXT and ROUTER edges explicitly when you want them in the graph.

```python
# What this shows: the three EdgeType values in use. TOOL is the
# one the framework auto-creates; CONTEXT and ROUTER are opt-in
# (usually added by custom engines or by explicit graph-authoring
# code).
# Why three kinds instead of one: renderers can colour / route
# them differently, and to_dict / from_dict round-trip the
# distinction. A ROUTER edge tells a reader "this is a decision
# branch", not "this is a callable dependency".

from lazybridge import Agent, Session, EdgeType

sess = Session()
# Note: nested agents passed as tools get EdgeType.TOOL edges only
# when they have NO session of their own at the time of wrapping —
# the outer session then propagates down and registers them. If you
# construct them with session=sess up-front you've opted into manual
# graph management; add any cross-edges you want explicitly.
researcher = Agent("claude-opus-4-7", name="researcher")      # no session →
writer     = Agent("claude-opus-4-7", name="writer")           # will inherit
monitor    = Agent("claude-opus-4-7", name="monitor", session=sess)

orchestrator = Agent(
    "claude-opus-4-7",
    name="orchestrator",
    tools=[researcher, writer],   # auto-creates TWO EdgeType.TOOL edges
                                  # (researcher/writer had no session yet)
    session=sess,
)

# A CONTEXT edge says "monitor observes orchestrator's state but
# isn't on its call path". Useful for sidebars / shadow agents that
# read a Session via sources= but aren't tools.
sess.graph.add_edge(
    monitor.name, orchestrator.name,
    label="observes", kind=EdgeType.CONTEXT,
)

# A ROUTER edge says "this is a branching decision". Typically
# emitted by custom routing engines; a declarative Plan's routing
# via output.next is represented by RouterNode, not loose edges.
# Shown here for completeness.
sess.graph.add_edge(
    orchestrator.name, writer.name,
    label="if publish=true", kind=EdgeType.ROUTER,
)
```

Edge addition is **idempotent** — adding the same `(from, to, label,
kind)` twice is a no-op, so re-registering agents across session
reuse doesn't bloat the graph.

## Router nodes — declaring branches as data

When a pipeline contains a *branching decision* (rather than a pure
call graph), the graph records it as a `RouterNode` — a distinct
node type that carries a `routes` dict (branch label → target node
id) plus an optional `default`.  Routers are usually **auto-registered
by Plan** when a step's `output.next: Literal[...]` field routes
across named steps; hand-registration is only needed when you're
writing a custom engine that wants its branching logic visible on the
graph.

```python
# What this shows: a custom engine exposing its router via the
# duck-type contract that GraphSchema.add_router accepts — any
# object whose ``to_graph_node()`` returns
#   {"id": ..., "name": ..., "routes": {...}, "default": ...}.
# Why duck-typed: LazyBridge's own Plan uses this contract
# internally; your engine implementing the same shape integrates
# automatically.

class MyRouter:
    def __init__(self, name):
        self.name = name
        self.routes = {
            "bug":     "researcher",
            "feature": "writer",
            "spam":    "monitor",
        }
        self.default = "researcher"

    def to_graph_node(self):
        return {
            "id": self.name,
            "name": self.name,
            "routes": self.routes,
            "default": self.default,
        }

sess.graph.add_router(MyRouter("intake_router"))
```

Serialisation preserves the `routes` dict so a saved `.yaml` / `.json`
graph carries the full branching logic — a renderer can draw it as a
diamond with labelled edges without needing the engine code.

## Round-trip: `save`, `from_file`, `clear`

GraphSchema exists to be **inspected** and **persisted**.  The
round-trip is descriptor-level — you get back node/edge metadata, not
live Agent instances — which is intentional: a saved graph is a
*topology*, not an executable program.

```python
# What this shows: saving a graph to disk, loading it in a different
# process, and clearing a session's graph for reuse inside a
# long-running worker.
# Why descriptor-only: serialising live Agents would require pinning
# model strings, provider creds, tool callables — brittle and
# environment-specific. Descriptors are stable across versions and
# processes.

# Persist. Extension decides format: .json uses stdlib json, .yaml
# needs PyYAML (pip install lazybridge[yaml]).
sess.graph.save("topology.yaml")

# Reconstruct in another process / test:
from lazybridge import GraphSchema
reloaded = GraphSchema.from_file("topology.yaml")
assert len(reloaded.nodes()) >= 3
assert {e.label for e in reloaded.edges()} >= {"observes"}

# For long-running workers that reuse the Session across requests,
# clear() wipes graph state while preserving session_id so new runs
# start with a clean topology.
sess.graph.clear()
assert len(sess.graph.nodes()) == 0
```

## Querying the graph: `edges_from` / `edges_to` / `node`

Once populated, the graph is a simple query target — useful for
debug panels, coverage assertions in tests, and custom topology
exports.

```python
# What this shows: assertions a test suite might make against the
# graph to catch "this orchestrator forgot to register X as a tool"
# before the bug reaches a user.
# Why the two directions: edges_from answers "who does X call?",
# edges_to answers "who calls X?". Both are O(edges) linear scans —
# GraphSchema is not a search index.

for e in sess.graph.edges_from(orchestrator.name):
    print(f"{orchestrator.name} -> {e.to_id} ({e.kind.value}: {e.label})")

callers = sess.graph.edges_to(researcher.name)
assert any(e.from_id == orchestrator.name for e in callers)

node = sess.graph.node(researcher.name)    # AgentNode or RouterNode
assert node is not None and node.type.value == "agent"
```

## Pitfalls

- An Agent without ``session=`` is not registered anywhere. If you pass
  it as a nested tool to an Agent with a session, the outer Agent
  propagates its session down and registers the nested one for you.
- ``to_yaml`` requires PyYAML (``pip install lazybridge[yaml]``);
  ``to_json`` is stdlib-only.
- ``from_dict`` reconstructs descriptors only — the ``provider`` /
  ``model`` strings on ``AgentNode`` are not live ``LLMEngine``s.

!!! note "API reference"

    GraphSchema(session_id: str = "") -> GraphSchema
    
    graph.add_agent(agent: Agent) -> None
    graph.add_router(router) -> None
    graph.add_edge(from_id, to_id, *, label="", kind=EdgeType.TOOL) -> None
    graph.nodes() -> list[_BaseNode]
    graph.edges() -> list[Edge]
    graph.edges_from(node_id) / edges_to(node_id) -> list[Edge]
    
    graph.to_dict() / to_json(indent=2) / to_yaml() -> str | dict
    GraphSchema.from_dict / from_json / from_file -> GraphSchema
    graph.save(path: str)     # .json or .yaml by extension
    
    NodeType (StrEnum):  AGENT, ROUTER
    EdgeType (StrEnum):  TOOL, CONTEXT, ROUTER
    
    Auto-populated: every Agent(session=s) registers into s.graph.
    Every as_tool wrapping records an edge with label="as_tool".

!!! warning "Rules & invariants"

    - Nodes are ``AgentNode`` (provider, model, system) or ``RouterNode``
      (routes, default). ``add_agent`` reads ``agent.id`` / ``name`` /
      ``engine.provider`` / ``engine.model`` (duck-typed).
    - ``session.register_tool_edge(outer, inner, label=…)`` adds an
      ``EdgeType.TOOL`` edge manually if you're wiring outside of
      ``as_tool`` (rare).
    - Serialisation is descriptor-only: reconstructing a runnable pipeline
      from a saved graph is the caller's job.

## See also

[session](session.md), [agent](agent.md),
[plan_serialize](plan-serialize.md)
