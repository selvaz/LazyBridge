## signature
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

## rules
- Nodes are ``AgentNode`` (provider, model, system) or ``RouterNode``
  (routes, default). ``add_agent`` reads ``agent.id`` / ``name`` /
  ``engine.provider`` / ``engine.model`` (duck-typed).
- ``session.register_tool_edge(outer, inner, label=…)`` adds an
  ``EdgeType.TOOL`` edge manually if you're wiring outside of
  ``as_tool`` (rare).
- Serialisation is descriptor-only: reconstructing a runnable pipeline
  from a saved graph is the caller's job.

## narrative
Two typical uses:

* **Debug** — `print(session.graph.to_json())` after a run to confirm
  the pipeline you built matches what ran.
* **External rendering** — `save("g.yaml")` → load in a diagramming
  tool or your own UI renderer.

## example
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

## pitfalls
- An Agent without ``session=`` is not registered anywhere. If you pass
  it as a nested tool to an Agent with a session, the outer Agent
  propagates its session down and registers the nested one for you.
- ``to_yaml`` requires PyYAML (``pip install lazybridge[yaml]``);
  ``to_json`` is stdlib-only.
- ``from_dict`` reconstructs descriptors only — the ``provider`` /
  ``model`` strings on ``AgentNode`` are not live ``LLMEngine``s.

