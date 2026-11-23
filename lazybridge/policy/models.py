"""Immutable policy value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PolicyError(ValueError):
    """Raised when a profile document or task grant is invalid."""


class PolicyWidening(PolicyError):
    """Raised when a delegated request exceeds its parent authority."""


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


@dataclass(frozen=True)
class EffectivePolicy:
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
    digest: str
