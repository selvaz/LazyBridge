"""Engine-neutral execution and delegation policy compiler.

The objects in this module are deliberately small, immutable value objects so
the same resolved policy can cross agent, tool-host, and process boundaries.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PolicyError(ValueError):
    """Base class for invalid policy configuration or requests."""


class PolicyWidening(PolicyError):
    """Raised when a delegated request exceeds its parent's authority."""


@dataclass(frozen=True)
class Profile:
    """A named, ordered ceiling on task permissions."""

    name: str
    tools: tuple[str, ...] = ()
    network: bool = False
    level: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise PolicyError("profile name must not be empty")
        if self.level < 0:
            raise PolicyError("profile level must not be negative")
        object.__setattr__(self, "tools", _normalise_tools(self.tools))


@dataclass(frozen=True)
class TaskGrant:
    """Task-specific request made against a profile."""

    profile: str | Profile
    folder: str | os.PathLike[str]
    tools: tuple[str, ...] | None = None
    network: bool | None = None

    def __post_init__(self) -> None:
        if not self.profile or (isinstance(self.profile, str) and not self.profile.strip()):
            raise PolicyError("grant profile must not be empty")
        if not os.fspath(self.folder):
            raise PolicyError("grant folder must not be empty")
        object.__setattr__(self, "folder", str(Path(self.folder).expanduser().resolve(strict=False)))
        if self.tools is not None:
            object.__setattr__(self, "tools", _normalise_tools(self.tools))


@dataclass(frozen=True)
class EffectivePolicy:
    """Fully resolved permissions, suitable for an engine compiler."""

    profile: Profile
    folder: str | os.PathLike[str]
    tools: tuple[str, ...]
    network: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "folder", str(Path(self.folder).expanduser().resolve(strict=False)))
        object.__setattr__(self, "tools", _normalise_tools(self.tools))


def _normalise_tools(tools: Any) -> tuple[str, ...]:
    if isinstance(tools, str):
        raise PolicyError("tools must be a sequence, not a string")
    values = tuple(tools)
    if any(not isinstance(tool, str) or not tool for tool in values):
        raise PolicyError("tool names must be non-empty strings")
    return tuple(sorted(set(values)))


def load_profiles(path: str | os.PathLike[str]) -> dict[str, Profile]:
    """Load profiles from TOML, rejecting ambiguous or malformed input."""

    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    raw = document.get("profiles")
    if not isinstance(raw, dict) or not raw:
        raise PolicyError("TOML must contain a non-empty [profiles] table")
    profiles: dict[str, Profile] = {}
    for name in sorted(raw):
        value = raw[name]
        if not isinstance(value, dict):
            raise PolicyError(f"profile {name!r} must be a table")
        unknown = set(value) - {"tools", "network", "level"}
        if unknown:
            raise PolicyError(f"profile {name!r} has unknown fields: {', '.join(sorted(unknown))}")
        profiles[name] = Profile(name, tuple(value.get("tools", ())), value.get("network", False), value.get("level", 0))
    return profiles


def resolve(profiles: Mapping[str, Profile], grant: TaskGrant) -> EffectivePolicy:
    """Resolve omitted grant fields to the selected profile's ceilings."""

    if isinstance(grant.profile, Profile):
        profile = grant.profile
    else:
        try:
            profile = profiles[grant.profile]
        except KeyError as exc:
            raise PolicyError(f"unknown profile: {grant.profile}") from exc
    tools = profile.tools if grant.tools is None else grant.tools
    extra = set(tools) - set(profile.tools)
    if extra:
        raise PolicyWidening(f"tool not granted by profile: {sorted(extra)[0]}")
    network = profile.network if grant.network is None else grant.network
    if network and not profile.network:
        raise PolicyWidening("network on under profile network off")
    return EffectivePolicy(profile, grant.folder, tools, network)


def compile_claude(policy: EffectivePolicy) -> dict[str, Any]:
    """Compile to deterministic Claude Code SDK option names."""

    return {
        "allowed_tools": list(policy.tools),
        "cwd": policy.folder,
        "network": policy.network,
        "permission_mode": "dontAsk",
    }


def compile_codex(policy: EffectivePolicy) -> dict[str, Any]:
    """Compile to deterministic Codex App Server option names."""

    return {
        "approval_policy": "never",
        "cwd": policy.folder,
        "network_access": policy.network,
        "sandbox_mode": "workspace-write",
        "tools": list(policy.tools),
    }


def child_within_parent(parent: EffectivePolicy, child_request: TaskGrant) -> EffectivePolicy:
    """Resolve a child grant while enforcing monotonic permission narrowing."""

    child_folder = Path(child_request.folder)
    parent_folder = Path(parent.folder)
    try:
        child_folder.relative_to(parent_folder)
    except ValueError as exc:
        raise PolicyWidening("folder outside parent folder") from exc
    child_profile = child_request.profile
    if isinstance(child_profile, str):
        if child_profile != parent.profile.name:
            # A name alone carries no permissions with which to prove narrowing.
            raise PolicyWidening("profile above parent")
        child_profile = parent.profile
    if child_profile.level > parent.profile.level:
        raise PolicyWidening("profile above parent")
    profile_tools = set(child_profile.tools)
    tools = tuple(tool for tool in parent.tools if tool in profile_tools) if child_request.tools is None else child_request.tools
    extra = set(tools) - set(parent.tools)
    if extra:
        raise PolicyWidening(f"tool not granted to parent: {sorted(extra)[0]}")
    network = parent.network and child_profile.network if child_request.network is None else child_request.network
    if network and not parent.network:
        raise PolicyWidening("network on under parent off")
    if network and not child_profile.network:
        raise PolicyWidening("network on under profile network off")
    return EffectivePolicy(child_profile, child_folder, tools, network)


def opposite_engine_reviewer(writer_engine: str) -> str:
    """Return the independent reviewer engine for a supported writer."""

    engines = {"claude": "codex", "claude-code": "codex", "codex": "claude"}
    try:
        reviewer = engines[writer_engine.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise PolicyError(f"unsupported writer engine: {writer_engine!r}") from exc
    if reviewer == writer_engine.strip().lower():  # defensive invariant
        raise PolicyError("same-engine review is not allowed")
    return reviewer


__all__ = [
    "EffectivePolicy",
    "PolicyError",
    "PolicyWidening",
    "Profile",
    "TaskGrant",
    "child_within_parent",
    "compile_claude",
    "compile_codex",
    "load_profiles",
    "opposite_engine_reviewer",
    "resolve",
]
