from dataclasses import replace

import pytest

from lazybridge.policy import (
    PolicyError,
    PolicyWidening,
    TaskGrant,
    child_within_parent,
    opposite_engine_reviewer,
    resolve,
)


def grant(root, **changes):
    values = {
        "tool_classes": ["read", "edit", "shell"],
        "tools": ["web", "mcp"],
        "network_routes": ["https://example.test"],
        "mcp_servers": {"server": {"tools": ["read", "write"]}},
        "mcp_inventory": {"server": {"command": "server", "tools": ["read", "write"]}},
        "scratch_folder": str(root / ".scratch"),
    }
    values.update(changes)
    return TaskGrant(root, values, "on", "local")


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"tool_classes": ["read", "edit", "shell"]}, "tool class"),
        ({"tools": ["web", "mcp"]}, "tool not granted"),
        ({"network_routes": ["https://other.test"]}, "network route"),
        ({"mcp_servers": {"server": {"tools": ["write"]}}}, "MCP server or tool"),
        ({"mcp_inventory": {"server": {"command": "other", "tools": ["read", "write"]}}}, "MCP server configuration"),
        ({"mcp_inventory": {}, "mcp_servers": {}}, "MCP inventory"),
    ],
)
def test_child_refuses_widening(tmp_path, change, reason):
    parent_grant = grant(tmp_path, tool_classes=["read"], tools=["mcp"], mcp_servers={"server": {"tools": ["read"]}})
    parent = resolve("edit-and-run", parent_grant)
    child_grant = replace(parent_grant, grants={**parent_grant.grants, **change})
    with pytest.raises(PolicyWidening, match=reason):
        child_within_parent(parent, child_grant)


def test_paths_network_placement_and_grandchild(tmp_path):
    parent = resolve("edit-and-run", grant(tmp_path))
    with pytest.raises(PolicyWidening, match="folder outside"):
        child_within_parent(parent, grant(tmp_path.parent, scratch_folder=str(tmp_path / ".scratch")))
    with pytest.raises(PolicyWidening, match="scratch folder"):
        child_within_parent(parent, grant(tmp_path, scratch_folder=str(tmp_path / "other")))
    with pytest.raises(PolicyWidening, match="placement"):
        child_within_parent(parent, replace(grant(tmp_path), placement="cloud"))
    offline = TaskGrant(
        tmp_path,
        {
            "tool_classes": ["read"],
            "scratch_folder": str(tmp_path / ".scratch"),
            "mcp_inventory": {"server": {"command": "server", "tools": ["read", "write"]}},
        },
        "off",
        "local",
    )
    child = child_within_parent(parent, offline)
    with pytest.raises(PolicyWidening, match="network on"):
        child_within_parent(child, replace(offline, network="on"))
    with pytest.raises(PolicyWidening, match="tool class"):
        child_within_parent(child, replace(offline, grants={**offline.grants, "tool_classes": ["edit"]}))


def test_scratch_and_alias_containment(tmp_path):
    with pytest.raises(PolicyError, match="scratch_folder"):
        resolve("read", TaskGrant(tmp_path, {"scratch_folder": "../escape"}, "off", "local"))
    parent = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    assert child_within_parent(parent, TaskGrant(tmp_path / ".", {}, "off", "local")).folder == parent.folder


def test_reviewer_is_always_opposite_engine():
    assert opposite_engine_reviewer("claude") == "codex"
    assert opposite_engine_reviewer("codex") == "claude"
    with pytest.raises(PolicyError, match="unsupported writer engine"):
        opposite_engine_reviewer("same")


def test_digest_is_canonical_for_unordered_grants(tmp_path):
    one = grant(tmp_path, network_routes=["https://b", "https://a"])
    two = grant(
        tmp_path,
        network_routes=["https://a", "https://b"],
        tool_classes=["shell", "read", "edit"],
        tools=["mcp", "web"],
    )
    assert resolve("edit-and-run", one).digest == resolve("edit-and-run", two).digest
