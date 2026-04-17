"""Unit tests for LazyRouter — T8.xx series."""

from __future__ import annotations

import pytest

from lazybridge.lazy_router import LazyRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, name: str, id: str = "abc") -> None:
        self.name = name
        self.id = id


# ---------------------------------------------------------------------------
# T8.01 — route() calls condition and returns correct agent
# ---------------------------------------------------------------------------


def test_route_returns_correct_agent():
    # T8.01
    a = _FakeAgent("writer")
    b = _FakeAgent("reviewer")
    router = LazyRouter(
        condition=lambda v: "writer" if v > 0.5 else "reviewer",
        routes={"writer": a, "reviewer": b},
    )
    assert router.route(0.9) is a
    assert router.route(0.1) is b


# ---------------------------------------------------------------------------
# T8.02 — route() raises KeyError for unknown key without default
# ---------------------------------------------------------------------------


def test_route_unknown_key_raises():
    # T8.02
    router = LazyRouter(
        condition=lambda v: "unknown",
        routes={"writer": _FakeAgent("writer")},
    )
    with pytest.raises(KeyError, match="unknown"):
        router.route("anything")


# ---------------------------------------------------------------------------
# T8.03 — default fallback used when key missing
# ---------------------------------------------------------------------------


def test_route_default_fallback():
    # T8.03
    fallback = _FakeAgent("fallback")
    router = LazyRouter(
        condition=lambda v: "nope",
        routes={"fallback": fallback},
        default="fallback",
    )
    assert router.route("x") is fallback


# ---------------------------------------------------------------------------
# T8.04 — condition returning non-str raises TypeError
# ---------------------------------------------------------------------------


def test_route_non_str_key_raises():
    # T8.04
    router = LazyRouter(
        condition=lambda v: 42,  # returns int, not str
        routes={"writer": _FakeAgent("writer")},
    )
    with pytest.raises(TypeError, match="must return a str key"):
        router.route("anything")


# ---------------------------------------------------------------------------
# T8.05 — aroute() with sync condition
# ---------------------------------------------------------------------------


async def test_aroute_sync_condition():
    # T8.05
    a = _FakeAgent("writer")
    router = LazyRouter(
        condition=lambda v: "writer",
        routes={"writer": a},
    )
    result = await router.aroute("x")
    assert result is a


# ---------------------------------------------------------------------------
# T8.06 — aroute() with async condition
# ---------------------------------------------------------------------------


async def test_aroute_async_condition():
    # T8.06
    b = _FakeAgent("reviewer")

    async def async_cond(v):
        return "reviewer"

    router = LazyRouter(
        condition=async_cond,
        routes={"reviewer": b},
    )
    result = await router.aroute("anything")
    assert result is b


# ---------------------------------------------------------------------------
# T8.07 — agent_names property
# ---------------------------------------------------------------------------


def test_agent_names():
    # T8.07
    a = _FakeAgent("alpha")
    b = _FakeAgent("beta")
    router = LazyRouter(
        condition=lambda v: "alpha",
        routes={"alpha": a, "beta": b},
    )
    names = router.agent_names
    assert "alpha" in names
    assert "beta" in names
    assert len(names) == 2


# ---------------------------------------------------------------------------
# T8.08 — to_graph_node() structure
# ---------------------------------------------------------------------------


def test_to_graph_node_structure():
    # T8.08
    a = _FakeAgent("writer", id="id-1")
    b = _FakeAgent("reviewer", id="id-2")
    router = LazyRouter(
        condition=lambda v: "writer",
        routes={"writer": a, "reviewer": b},
        name="quality_gate",
        default="reviewer",
    )
    node = router.to_graph_node()
    assert node["type"] == "router"
    assert node["name"] == "quality_gate"
    assert node["default"] == "reviewer"
    assert "writer" in node["routes"]
    assert "reviewer" in node["routes"]


# ---------------------------------------------------------------------------
# T8.09 — repr includes name and route keys
# ---------------------------------------------------------------------------


def test_repr():
    # T8.09
    router = LazyRouter(
        condition=lambda v: "a",
        routes={"a": _FakeAgent("a"), "b": _FakeAgent("b")},
        name="my_router",
    )
    r = repr(router)
    assert "my_router" in r
    assert "a" in r


# ---------------------------------------------------------------------------
# T8.10 — default set but unknown — falls back correctly
# ---------------------------------------------------------------------------


def test_default_only_used_when_key_missing():
    # T8.10
    primary = _FakeAgent("primary")
    fallback = _FakeAgent("fallback")
    router = LazyRouter(
        condition=lambda v: "primary" if v else "missing",
        routes={"primary": primary, "fallback": fallback},
        default="fallback",
    )
    assert router.route(True) is primary  # key exists → no fallback
    assert router.route(False) is fallback  # key missing → use default
