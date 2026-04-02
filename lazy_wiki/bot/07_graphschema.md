### GraphSchema

Directed graph of agents, routers, and their connections. Created automatically by `LazySession`. Can be serialized to JSON/YAML for GUI loading.

```python
GraphSchema(session_id: str)
```

### Node types

```python
class NodeType(StrEnum):
    AGENT  = "agent"
    ROUTER = "router"
```

`AgentNode`: `{type, id, name, provider, model}`
`RouterNode`: `{type, id, name, routes: {key: agent_id}, default}`

### Edge types

```python
class EdgeType(StrEnum):
    TOOL    = "tool"      # agent A calls agent B as a tool
    CONTEXT = "context"   # agent A reads agent B's output via LazyContext.from_agent
    ROUTER  = "router"    # conditional branch from a LazyRouter
```

### Mutation

```python
schema.add_agent(agent: LazyAgent) -> None
```
Called automatically by `LazySession._register_agent()`. Adds `AgentNode`.

```python
schema.add_router(router: LazyRouter) -> None
```
Adds `RouterNode`. Call manually after constructing a `LazyRouter`.

```python
schema.add_edge(
    src_id: str,
    dst_id: str,
    type: EdgeType | str = EdgeType.TOOL,
    label: str | None = None,
) -> None
```
Adds a directed edge. `src_id` and `dst_id` are node IDs (agent.id or router.name).

### Serialization

```python
schema.to_dict() -> dict
schema.to_json(indent: int = 2) -> str
schema.to_yaml() -> str                    # requires PyYAML
schema.save(path: str | Path) -> None      # auto-detects .json / .yaml / .yml
```

JSON structure:
```json
{
  "session_id": "uuid",
  "nodes": {
    "agent-uuid": {
      "type": "agent",
      "id": "agent-uuid",
      "name": "researcher",
      "provider": "anthropic",
      "model": "claude-sonnet-4-6"
    }
  },
  "edges": [
    {"src": "orchestrator-uuid", "dst": "researcher-uuid", "type": "tool", "label": "researcher"}
  ]
}
```

### Deserialization

```python
GraphSchema.from_dict(data: dict) -> GraphSchema
GraphSchema.from_json(text: str) -> GraphSchema
GraphSchema.from_file(path: str | Path) -> GraphSchema   # auto-detects .json / .yaml
```
Note: deserialization reconstructs the graph descriptor only — not live agent instances. Reconstruction of live agents from JSON is future work.

### Usage

```python
from lazybridge import LazyAgent, LazySession

sess = LazySession()
a1 = LazyAgent("anthropic", name="researcher", session=sess)
a2 = LazyAgent("openai",    name="writer",     session=sess)

# agents auto-registered; add tool edge manually
sess.graph.add_edge(a1.id, a2.id, type="context", label="researcher→writer")

# inspect
print(sess.graph.to_json())

# save for GUI
sess.graph.save("pipeline.json")
sess.graph.save("pipeline.yaml")

# reload (descriptor only)
loaded = GraphSchema.from_file("pipeline.json")
```
