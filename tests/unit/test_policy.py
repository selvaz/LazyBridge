from dataclasses import replace

import pytest

from lazybridge.policy import (
    PolicyBudgetExceeded,
    PolicyError,
    PolicyWidening,
    TaskGrant,
    TreeBudget,
    child_within_parent,
    opposite_engine_reviewer,
    policy_scope,
    resolve,
    select_reviewer,
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


@pytest.mark.parametrize("scratch_name", [None, "custom"])
def test_read_only_child_can_narrow_to_subfolder(tmp_path, scratch_name):
    parent = resolve("edit", TaskGrant(tmp_path, {"tool_classes": ["read", "edit"]}, "off", "local"))
    grants = {"tool_classes": ["read"]}
    if scratch_name is not None:
        grants["scratch_folder"] = str(parent.scratch_folder / scratch_name)
    child = child_within_parent(parent, TaskGrant(tmp_path / "sub", grants, "off", "local"), profile="read")

    assert child.folder == parent.folder / "sub"
    assert child.scratch_folder == parent.scratch_folder / (scratch_name or "sub")
    assert child.tool_classes == {"read"}


def test_child_keeps_default_scratch_already_in_parent_scratch(tmp_path):
    parent = resolve("read", TaskGrant(tmp_path, {"scratch_folder": "."}, "off", "local"))
    child = child_within_parent(parent, TaskGrant(tmp_path / "sub", {}, "off", "local"))

    assert child.scratch_folder == child.folder / ".scratch"


def test_subfolder_child_cannot_widen_tool_classes(tmp_path):
    parent = resolve("edit", TaskGrant(tmp_path, {"tool_classes": ["read", "edit"]}, "off", "local"))
    request = TaskGrant(tmp_path / "sub", {"tool_classes": ["read", "shell"]}, "off", "local")

    with pytest.raises(PolicyWidening, match="tool class not granted to parent"):
        child_within_parent(parent, request)


@pytest.mark.parametrize("scratch", [".scratch", "../.scratch/../escape"])
def test_subfolder_child_cannot_escape_parent_scratch(tmp_path, scratch):
    parent = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    request = TaskGrant(tmp_path / "sub", {"scratch_folder": scratch}, "off", "local")

    with pytest.raises(PolicyWidening, match="scratch folder outside parent scratch folder"):
        child_within_parent(parent, request)


def test_reviewer_is_always_opposite_engine():
    assert opposite_engine_reviewer("claude") == "codex"
    assert opposite_engine_reviewer("codex") == "claude"
    with pytest.raises(PolicyError, match="unsupported writer engine"):
        opposite_engine_reviewer("same")


def test_default_reviewer_selection_refuses_same_engine():
    assert select_reviewer("claude", {"codex", "claude"}) == ("codex", "opposite-engine")
    with pytest.raises(PolicyError, match="opposite reviewer engine 'codex' is unavailable"):
        select_reviewer("claude", {"claude"})


@pytest.mark.parametrize(
    "writer,available,expected",
    [
        ("claude", ["claude", "codex"], ("codex", "opposite-engine")),
        ("codex", ["codex", "Claude-Code"], ("claude", "opposite-engine")),
        ("Claude-Code", ["claude", "claude-code"], ("claude", "same-provider-fresh")),
        ("codex", ["codex"], ("codex", "same-provider-fresh")),
    ],
)
def test_fallback_reviewer_selection(writer, available, expected):
    assert select_reviewer(writer, iter(available), "opposite-or-fresh") == expected


@pytest.mark.parametrize("available", [[], ["claude", "other"], ["other"]])
def test_fallback_reviewer_requires_the_writer_engine_to_be_the_only_available_engine(available):
    with pytest.raises(PolicyError, match="unavailable"):
        select_reviewer("claude", available, "opposite-or-fresh")


def test_reviewer_selection_refuses_unknown_mode():
    with pytest.raises(PolicyError, match="unsupported review mode"):
        select_reviewer("claude", ["codex"], "unknown")


def test_digest_is_canonical_for_unordered_grants(tmp_path):
    one = grant(tmp_path, network_routes=["https://b", "https://a"])
    two = grant(
        tmp_path,
        network_routes=["https://a", "https://b"],
        tool_classes=["shell", "read", "edit"],
        tools=["mcp", "web"],
    )
    assert resolve("edit-and-run", one).digest == resolve("edit-and-run", two).digest


def test_digest_changes_with_inventory_tools(tmp_path):
    one = grant(tmp_path)
    two = grant(tmp_path, mcp_inventory={"server": {"command": "server", "tools": ["read", "write", "delete"]}})

    assert resolve("edit-and-run", one).digest != resolve("edit-and-run", two).digest


def test_tree_budget_depth_and_named_error(tmp_path):
    authority = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    budget = TreeBudget(max_depth=1)
    with (
        policy_scope(authority, tree_budget=budget),
        policy_scope(None, child_id="child"),
        pytest.raises(PolicyBudgetExceeded) as raised,
        policy_scope(None, child_id="grandchild"),
    ):
        pass
    assert raised.value.bound == "max_depth"
    assert "max_depth" in str(raised.value)


def test_tree_budget_total_and_resume_slot_reuse(tmp_path):
    authority = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    budget = TreeBudget(max_total_children=1, max_work_units=1)
    with policy_scope(authority, tree_budget=budget):
        with policy_scope(None, child_id="stable-session"):
            pass
        # Re-admitting the same identity is a resume and debits nothing new.
        with policy_scope(None, child_id="stable-session"):
            pass
        with (
            pytest.raises(PolicyBudgetExceeded, match="max_total_children"),
            policy_scope(None, child_id="different-session"),
        ):
            pass


def test_tree_budget_reserves_active_review_slot(tmp_path):
    authority = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    budget = TreeBudget(max_active_children=2)
    with (
        policy_scope(authority, tree_budget=budget),
        policy_scope(None, child_id="writer", reserve_review=True, engine="claude"),
    ):
        with pytest.raises(PolicyBudgetExceeded) as raised, policy_scope(None, child_id="ordinary"):
            pass
        with (
            pytest.raises(PolicyBudgetExceeded),
            policy_scope(None, child_id="wrong-reviewer", role="reviewer", engine="claude", fresh=True),
        ):
            pass
        with (
            policy_scope(None, child_id="reviewer", role="reviewer", engine="codex") as context,
            pytest.raises(PolicyBudgetExceeded),
            policy_scope(None, child_id="extra"),
        ):
            pass
        assert context.review_independence == "opposite-engine"
    assert raised.value.bound == "max_active_children"


def test_fallback_reserves_for_the_opposite_engine_when_available():
    budget = TreeBudget(review_mode="opposite-or-fresh")
    budget.admit("writer", depth=1, reserve_review=True, engine="claude")

    with pytest.raises(PolicyError, match="single available engine"):
        budget.admit("same", depth=1, role="reviewer", engine="claude", fresh=True)

    assert budget.admit("opposite", depth=1, role="reviewer", engine="codex") == "opposite-engine"


def test_fallback_reservation_refuses_a_non_fresh_or_reused_reviewer():
    budget = TreeBudget(max_total_children=3, review_mode="opposite-or-fresh", available_engines={"codex"})
    budget.admit("reused", depth=1)
    budget.release("reused")
    budget.admit("writer", depth=1, reserve_review=True, engine="codex")

    with pytest.raises(PolicyError, match="fresh reviewer"):
        budget.admit("not-fresh", depth=1, role="reviewer", engine="codex")
    with pytest.raises(PolicyError, match="fresh reviewer"):
        budget.admit("reused", depth=1, role="reviewer", engine="codex", fresh=True)
    with pytest.raises(PolicyError, match="fresh reviewer"):
        budget.admit("writer", depth=1, role="reviewer", engine="codex", fresh=True)

    assert budget.admit("fresh", depth=1, role="reviewer", engine="codex", fresh=True) == "same-provider-fresh"


def test_fallback_reservation_protects_active_slot_until_a_fresh_reviewer_claims_it():
    budget = TreeBudget(max_active_children=2, review_mode="opposite-or-fresh", available_engines={"claude"})
    with (
        policy_scope(None, tree_budget=budget),
        policy_scope(None, child_id="writer", reserve_review=True, engine="claude"),
    ):
        with pytest.raises(PolicyBudgetExceeded, match="max_active_children"):
            budget.admit("other", depth=1, role="reviewer", engine="codex", fresh=True)
        with pytest.raises(PolicyBudgetExceeded, match="max_active_children"):
            budget.admit("not-fresh", depth=1, role="reviewer", engine="claude")
        with policy_scope(None, child_id="fresh", role="reviewer", engine="claude", fresh=True) as context:
            assert context.review_independence == "same-provider-fresh"
            with pytest.raises(PolicyBudgetExceeded, match="max_active_children"):
                budget.admit("extra", depth=1)
        assert budget.admit("extra", depth=1) is None


def test_review_reservation_survives_writer_exit(tmp_path):
    authority = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    budget = TreeBudget(max_active_children=1)
    # A one-slot budget cannot admit a writer plus a review reservation.
    with (
        policy_scope(authority, tree_budget=budget),
        pytest.raises(PolicyBudgetExceeded),
        policy_scope(None, child_id="writer", reserve_review=True, engine="claude"),
    ):
        pass

    budget = TreeBudget(max_active_children=2)
    with policy_scope(authority, tree_budget=budget):
        with policy_scope(None, child_id="writer", reserve_review=True, engine="claude"):
            pass
        with policy_scope(None, child_id="ordinary"):
            with pytest.raises(PolicyBudgetExceeded), policy_scope(None, child_id="other"):
                pass
            with policy_scope(None, child_id="reviewer", role="reviewer", engine="codex"):
                pass


def test_tree_budget_work_unit_error_names_bound(tmp_path):
    authority = resolve("read", TaskGrant(tmp_path, {}, "off", "local"))
    budget = TreeBudget(max_work_units=2)
    with policy_scope(authority, tree_budget=budget), policy_scope(None, child_id="small", work_units=2):
        pass
    with (
        policy_scope(authority, tree_budget=budget),
        pytest.raises(PolicyBudgetExceeded) as raised,
        policy_scope(None, child_id="extra", work_units=1),
    ):
        pass
    assert raised.value.bound == "max_work_units"
