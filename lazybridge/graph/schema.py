"""GraphSchema — serialisable representation of an agent pipeline.

A GraphSchema is a directed graph where:
  - Nodes are LazyAgents or LazyRouters
  - Edges are LazyTools (agent → agent delegation) or LazyContext reads

The schema is auto-built as agents and tools are created, and can be:
  - Serialised to JSON/YAML for storage or GUI loading
  - Deserialised back into a descriptor via from_json() / from_file()
  - Inspected programmatically (for debugging or visualisation)

Note: deserialisation reconstructs the graph descriptor (nodes, edges, metadata)
but not live agent instances. Recreating runnable LazyAgent objects from the
schema is the caller's responsibility.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Node and Edge types
# ---------------------------------------------------------------------------

class NodeType(StrEnum):
    AGENT  = "agent"
    ROUTER = "router"


class EdgeType(StrEnum):
    TOOL    = "tool"      # agent A calls agent B as a tool
    CONTEXT = "context"   # agent A reads agent B's output via LazyContext
    ROUTER  = "router"    # conditional branch from a LazyRouter


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_yaml_path(path: str) -> bool:
    """Return True if the path has a YAML extension."""
    return path.endswith(".yaml") or path.endswith(".yml")


def _require_yaml() -> Any:
    """Import and return the yaml module, raising a clear error if missing."""
    try:
        import yaml  # type: ignore[import-untyped]
        return yaml
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")


# ---------------------------------------------------------------------------
# Node / Edge descriptors
# ---------------------------------------------------------------------------

class _BaseNode:
    """Shared attributes for all graph nodes."""
    __slots__ = ("id", "name", "type")

    def __init__(self, id: str, name: str) -> None:
        self.id = id
        self.name = name


class AgentNode(_BaseNode):
    """Serialisable descriptor for a LazyAgent node."""

    __slots__ = ("provider", "model", "system")

    def __init__(
        self,
        id: str,
        name: str,
        provider: str = "",
        model: str = "",
        system: str | None = None,
    ) -> None:
        super().__init__(id, name)
        self.provider = provider
        self.model = model
        self.system = system
        self.type = NodeType.AGENT

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "name":     self.name,
            "provider": self.provider,
            "model":    self.model,
            "system":   self.system,
            "type":     self.type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AgentNode:
        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            system=d.get("system"),
        )


class RouterNode(_BaseNode):
    """Serialisable descriptor for a LazyRouter node."""

    __slots__ = ("routes", "default")

    def __init__(self, id: str, name: str, routes: dict[str, str], default: str | None) -> None:
        super().__init__(id, name)
        self.routes = routes  # label → target_node_id
        self.default = default
        self.type = NodeType.ROUTER

    def to_dict(self) -> dict:
        return {
            "id":      self.id,
            "name":    self.name,
            "routes":  self.routes,
            "default": self.default,
            "type":    self.type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RouterNode:
        return cls(
            id=d["id"],
            name=d.get("name", d["id"]),
            routes=d.get("routes", {}),
            default=d.get("default"),
        )


class Edge:
    """A directed connection between two nodes."""

    __slots__ = ("from_id", "to_id", "label", "kind")

    def __init__(
        self,
        from_id: str,
        to_id: str,
        label: str = "",
        kind: EdgeType = EdgeType.TOOL,
    ) -> None:
        self.from_id = from_id
        self.to_id = to_id
        self.label = label
        self.kind = kind

    def to_dict(self) -> dict:
        return {
            "from":  self.from_id,
            "to":    self.to_id,
            "label": self.label,
            "type":  self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(
            from_id=d["from"],
            to_id=d["to"],
            label=d.get("label", ""),
            kind=EdgeType(d.get("type", "tool")),
        )


# ---------------------------------------------------------------------------
# GraphSchema
# ---------------------------------------------------------------------------

class GraphSchema:
    """Directed graph of agents, routers, and their connections.

    Auto-populated by LazySession._register_agent() as agents are created.
    Can also be built manually for GUI-driven pipeline construction.
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self._nodes: dict[str, _BaseNode] = {}
        self._edges: list[Edge] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_agent(self, agent: Any) -> None:
        """Register a LazyAgent as a graph node."""
        node = AgentNode(
            id=agent.id,
            name=getattr(agent, "name", agent.id),
            provider=getattr(agent, "_provider_name", ""),
            model=getattr(agent, "_model_name", ""),
            system=getattr(agent, "system", None),
        )
        self._nodes[node.id] = node

    def add_router(self, router: Any) -> None:
        """Register a LazyRouter as a graph node."""
        node_dict = router.to_graph_node()
        node = RouterNode(
            id=node_dict.get("id", node_dict.get("name", "")),
            name=node_dict.get("name", ""),
            routes=node_dict.get("routes", {}),
            default=node_dict.get("default"),
        )
        self._nodes[node.id] = node

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        *,
        label: str = "",
        kind: EdgeType | str = EdgeType.TOOL,
    ) -> None:
        self._edges.append(Edge(
            from_id=from_id,
            to_id=to_id,
            label=label,
            kind=EdgeType(kind) if isinstance(kind, str) else kind,
        ))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def node(self, node_id: str) -> _BaseNode | None:
        return self._nodes.get(node_id)

    def nodes(self) -> list[_BaseNode]:
        return list(self._nodes.values())

    def edges(self) -> list[Edge]:
        return list(self._edges)

    def edges_from(self, node_id: str) -> list[Edge]:
        return [e for e in self._edges if e.from_id == node_id]

    def edges_to(self, node_id: str) -> list[Edge]:
        return [e for e in self._edges if e.to_id == node_id]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "nodes":      [n.to_dict() for n in self._nodes.values()],  # type: ignore[attr-defined]
            "edges":      [e.to_dict() for e in self._edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        yaml = _require_yaml()
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    @classmethod
    def from_dict(cls, data: dict) -> GraphSchema:
        g = cls(session_id=data.get("session_id", ""))
        for n in data.get("nodes", []):
            if n.get("type") == NodeType.ROUTER:
                g._nodes[n["id"]] = RouterNode.from_dict(n)
            else:
                g._nodes[n["id"]] = AgentNode.from_dict(n)
        g._edges = [Edge.from_dict(e) for e in data.get("edges", [])]
        return g

    @classmethod
    def from_json(cls, text: str) -> GraphSchema:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: str) -> GraphSchema:
        with open(path) as f:
            text = f.read()
        if _is_yaml_path(path):
            yaml = _require_yaml()
            return cls.from_dict(yaml.safe_load(text))
        return cls.from_json(text)

    def save(self, path: str) -> None:
        """Save schema to JSON or YAML depending on file extension."""
        with open(path, "w") as f:
            f.write(self.to_yaml() if _is_yaml_path(path) else self.to_json())

    def __repr__(self) -> str:
        return f"GraphSchema(session={self.session_id[:8]}..., nodes={len(self._nodes)}, edges={len(self._edges)})"
