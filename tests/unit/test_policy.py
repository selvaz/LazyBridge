from pathlib import Path

import pytest

from lazybridge.policy import (
    EffectivePolicy,
    PolicyError,
    PolicyWidening,
    Profile,
    TaskGrant,
    child_within_parent,
    compile_claude,
    compile_codex,
    load_profiles,
    opposite_engine_reviewer,
    resolve,
)


def _parent(tmp_path: Path, *, network: bool = False) -> EffectivePolicy:
    profile = Profile("workspace", ("Read", "Write"), network, 2)
    return EffectivePolicy(profile, tmp_path, profile.tools, network)


def test_load_resolve_and_compile_example() -> None:
    path = Path(__file__).parents[2] / "lazybridge/policy/data/profiles.example.toml"
    profiles = load_profiles(path)
    policy = resolve(profiles, TaskGrant("workspace", ".", ("Read",), False))
    assert policy.tools == ("Read",)
    assert compile_claude(policy)["allowed_tools"] == ["Read"]
    assert compile_codex(policy)["network_access"] is False


@pytest.mark.parametrize(
    ("grant_factory", "reason"),
    [
        (lambda root: TaskGrant("workspace", root.parent), "folder outside parent folder"),
        (lambda root: TaskGrant("workspace", root, ("Shell",)), "tool not granted to parent"),
        (lambda root: TaskGrant("workspace", root, network=True), "network on under parent off"),
        (lambda root: TaskGrant(Profile("admin", ("Read",), False, 3), root), "profile above parent"),
    ],
)
def test_child_refuses_each_widening(tmp_path: Path, grant_factory, reason: str) -> None:
    with pytest.raises(PolicyWidening, match=reason):
        child_within_parent(_parent(tmp_path), grant_factory(tmp_path))


def test_child_can_narrow_every_dimension(tmp_path: Path) -> None:
    child_dir = tmp_path / "child"
    child = Profile("reader", ("Read",), False, 1)
    actual = child_within_parent(_parent(tmp_path, network=True), TaskGrant(child, child_dir))
    assert actual == EffectivePolicy(child, child_dir, ("Read",), False)


def test_reviewer_is_always_opposite_engine() -> None:
    assert opposite_engine_reviewer("claude") == "codex"
    assert opposite_engine_reviewer("codex") == "claude"
    with pytest.raises(PolicyError, match="unsupported writer engine"):
        opposite_engine_reviewer("same")
