"""Immutable policy value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cards import ApprovalCard
from .errors import PolicyDenied as PolicyDenied
from .errors import PolicyError as PolicyError
from .errors import PolicyWidening as PolicyWidening


@dataclass(frozen=True)
class Profile:
    name: str
    allowed_tool_classes: frozenset[str]


@dataclass(frozen=True)
class ProfileDocument:
    version: int
    revision: str
    profiles: Mapping[str, Profile]


@dataclass(frozen=True)
class TaskGrant:
    folder: Path | str
    grants: Mapping[str, Any]
    network: str
    placement: str
    approval_authority: frozenset[str] = frozenset()
    approval_cards: tuple[ApprovalCard, ...] = ()


@dataclass(frozen=True)
class EffectivePolicy:
    """Effective authority and diagnostics about grants removed by the profile ceiling."""

    profile: str
    version: int
    revision: str
    folder: Path
    scratch_folder: Path
    tool_classes: frozenset[str]
    tools: frozenset[str]
    network: str
    network_routes: tuple[str, ...]
    placement: str
    mcp_servers: Mapping[str, Mapping[str, Any]]
    mcp_inventory: tuple[str, ...]
    mcp_inventory_tools: Mapping[str, tuple[str, ...]]
    digest: str
    removed_grants: tuple[tuple[str, str], ...] = field(default=(), compare=False)
    _document: ProfileDocument | None = field(default=None, compare=False, repr=False)
    approval_authority: frozenset[str] = frozenset()
    approval_cards: tuple[ApprovalCard, ...] = ()
