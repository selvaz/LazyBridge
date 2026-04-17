"""Unit tests for GraphSchema — T11.xx series."""

from __future__ import annotations

from lazybridge.graph.schema import AgentNode, EdgeType, GraphSchema, NodeType, RouterNode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, name: str, id: str, provider: str = "anthropic", model: str = "m") -> None:
        self.id = id
        self.name = name
        self._provider_name = provider
        self._model_name = model
        self.system = None


class _FakeRouter:
    def __init__(self, name: str = "router") -> None:
        self.name = name

    def to_graph_node(self) -> dict:
        return {
            "type": "router",
            "name": self.name,
            "id": self.name,
            "routes": {"a": "id-a", "b": "id-b"},
            "default": None,
        }


# ---------------------------------------------------------------------------
# T11.01 — add_agent registers node
# ---------------------------------------------------------------------------


def test_add_agent_registers_node():
    # T11.01
    g = GraphSchema("sess-1")
    agent = _FakeAgent("researcher", "id-1")
    g.add_agent(agent)
    assert g.node("id-1") is not None
    assert g.node("id-1").name == "researcher"
    assert isinstance(g.node("id-1"), AgentNode)


# ---------------------------------------------------------------------------
# T11.02 — add_router registers RouterNode
# ---------------------------------------------------------------------------


def test_add_router_registers_node():
    # T11.02
    g = GraphSchema("sess-2")
    router = _FakeRouter("brancher")
    g.add_router(router)
    node = g.node("brancher")
    assert node is not None
    assert isinstance(node, RouterNode)
    assert node.name == "brancher"


# ---------------------------------------------------------------------------
# T11.03 — add_edge stores edge
# ---------------------------------------------------------------------------


def test_add_edge_stored():
    # T11.03
    g = GraphSchema("sess-3")
    g.add_edge("id-a", "id-b", kind=EdgeType.TOOL, label="calls")
    edges = g.edges()
    assert len(edges) == 1
    assert edges[0].from_id == "id-a"
    assert edges[0].to_id == "id-b"
    assert edges[0].kind == EdgeType.TOOL
    assert edges[0].label == "calls"


# ---------------------------------------------------------------------------
# T11.04 — to_dict includes session_id, nodes, edges
# ---------------------------------------------------------------------------


def test_to_dict_structure():
    # T11.04
    g = GraphSchema("sess-4")
    g.add_agent(_FakeAgent("a", "id-a"))
    g.add_agent(_FakeAgent("b", "id-b"))
    g.add_edge("id-a", "id-b", kind=EdgeType.CONTEXT)
    d = g.to_dict()
    assert d["session_id"] == "sess-4"
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1


# ---------------------------------------------------------------------------
# T11.05 — to_json / from_json round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip():
    # T11.05
    g = GraphSchema("sess-5")
    g.add_agent(_FakeAgent("researcher", "id-r", provider="anthropic", model="claude"))
    g.add_agent(_FakeAgent("writer", "id-w", provider="openai", model="gpt"))
    g.add_edge("id-r", "id-w", kind=EdgeType.TOOL, label="handoff")

    text = g.to_json()
    g2 = GraphSchema.from_json(text)

    assert g2.session_id == "sess-5"
    assert len(g2.nodes()) == 2
    assert len(g2.edges()) == 1
    assert g2.node("id-r").name == "researcher"
    assert g2.edges()[0].kind == EdgeType.TOOL


# ---------------------------------------------------------------------------
# T11.06 — from_dict restores RouterNode by type field
# ---------------------------------------------------------------------------


def test_from_dict_restores_router_node():
    # T11.06
    data = {
        "session_id": "sess-6",
        "nodes": [
            {"id": "r1", "name": "brancher", "type": "router", "routes": {}, "default": None},
        ],
        "edges": [],
    }
    g = GraphSchema.from_dict(data)
    node = g.node("r1")
    assert isinstance(node, RouterNode)
    assert node.type == NodeType.ROUTER


# ---------------------------------------------------------------------------
# T11.07 — edges_from / edges_to filter correctly
# ---------------------------------------------------------------------------


def test_edges_from_and_to():
    # T11.07
    g = GraphSchema()
    g.add_edge("a", "b", kind=EdgeType.TOOL)
    g.add_edge("a", "c", kind=EdgeType.CONTEXT)
    g.add_edge("b", "c", kind=EdgeType.TOOL)

    from_a = g.edges_from("a")
    assert len(from_a) == 2

    to_c = g.edges_to("c")
    assert len(to_c) == 2

    to_a = g.edges_to("a")
    assert len(to_a) == 0


# ---------------------------------------------------------------------------
# T11.08 — save / from_file round-trip (JSON)
# ---------------------------------------------------------------------------


def test_save_and_load_json(tmp_path):
    # T11.08
    path = str(tmp_path / "graph.json")
    g = GraphSchema("sess-8")
    g.add_agent(_FakeAgent("agent", "id-1"))
    g.save(path)

    g2 = GraphSchema.from_file(path)
    assert g2.session_id == "sess-8"
    assert g2.node("id-1").name == "agent"
