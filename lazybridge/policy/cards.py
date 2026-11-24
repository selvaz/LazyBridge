"""Immutable approval cards and deterministic matching for direct commands."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from .models import EffectivePolicy

ApprovalTier = Literal["routine", "sensitive", "external"]
CardTier = Literal["routine", "sensitive"]
NetworkState = Literal["off", "on"]

_EXECUTABLE = re.compile(r"[a-zA-Z0-9_.-]+")
_SHELL_SYNTAX = re.compile(r"""[;&|<>()`$%!^*?\[\]{}'"\x00-\x1f\x7f]""")
_COMMAND_WRAPPERS = frozenset(
    {
        "bash",
        "cmd",
        "dash",
        "env",
        "eval",
        "exec",
        "fish",
        "ksh",
        "powershell",
        "pwsh",
        "sh",
        "sudo",
        "wsl",
        "xargs",
        "zsh",
    }
)


class CardMatcher(BaseModel):
    """A fixed executable and subcommand, optionally restricted to exact arguments."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    command: tuple[str, ...] = Field(min_length=1)
    arguments: tuple[str, ...] | None = None
    branch: str | None = None
    path_prefix: Path | None = None
    network: NetworkState = "off"

    @model_validator(mode="after")
    def validate_matcher(self) -> CardMatcher:
        if not _single_command((*self.command, *(self.arguments or ()))):
            raise ValueError("card matcher requires a single direct command")
        if self.path_prefix is not None and not self.path_prefix.is_absolute():
            raise ValueError("card path prefix must be absolute")
        return self


class ApprovalCard(BaseModel):
    """A standing grant whose provenance contains every upstream card ID."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    card_id: str
    action_class: str = Field(min_length=1)
    tier: CardTier
    matcher: CardMatcher
    expires_at: AwareDatetime | None = None
    max_uses: int | None = Field(default=None, gt=0, strict=True)
    task_id: str | None = Field(default=None, min_length=1)
    project_id: str | None = Field(default=None, min_length=1)
    transferable: bool = False
    provenance: tuple[str, ...] = ()

    @property
    def chain(self) -> tuple[str, ...]:
        return (*self.provenance, self.card_id)


def _single_command(argv: tuple[str, ...]) -> bool:
    if not argv:
        return False
    executable = argv[0].lower()
    return (
        _EXECUTABLE.fullmatch(argv[0]) is not None
        and argv[0] not in {".", ".."}
        and executable.removesuffix(".exe").removesuffix(".com") not in _COMMAND_WRAPPERS
        and not executable.endswith((".bat", ".cmd", ".ps1"))
        and all(_SHELL_SYNTAX.search(arg) is None for arg in argv)
    )


def validate_card_copy(parent: ApprovalCard, child: ApprovalCard) -> None:
    """Require a direct copy that narrows or retains each restriction."""
    from .models import PolicyWidening

    if (
        not parent.transferable
        or child.provenance != parent.chain
        or child.card_id in parent.chain
        or child.action_class != parent.action_class
        or child.tier != parent.tier
    ):
        raise PolicyWidening("card is not a transferable copy of a card held by the parent")
    source, copy = parent.matcher, child.matcher
    if (
        copy.command[: len(source.command)] != source.command
        or copy.network != source.network
        or (
            source.arguments is not None
            and (copy.arguments is None or (*copy.command, *copy.arguments) != (*source.command, *source.arguments))
        )
        or (source.branch is not None and copy.branch != source.branch)
        or (
            source.path_prefix is not None
            and (copy.path_prefix is None or not copy.path_prefix.is_relative_to(source.path_prefix))
        )
    ):
        raise PolicyWidening("card matcher widens the parent card")
    if (
        (parent.project_id is not None and child.project_id != parent.project_id)
        or (parent.task_id is not None and child.task_id != parent.task_id)
        or (parent.max_uses is not None and (child.max_uses is None or child.max_uses > parent.max_uses))
        or (parent.expires_at is not None and (child.expires_at is None or child.expires_at > parent.expires_at))
    ):
        raise PolicyWidening("card limits widen the parent card")


class CardAction(BaseModel):
    """Trusted execution metadata for one argv invocation, with every affected path."""

    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    argv: tuple[str, ...]
    cwd: Path
    paths: tuple[Path, ...]
    network: NetworkState
    branch: str | None = None
    project_id: str | None = None


def _canonical_path(path: Path, base: Path) -> Path:
    if not path.is_absolute() and (path.drive or path.root):
        raise ValueError("ambiguous path")
    if isinstance(path, PureWindowsPath):
        parts = path.parts[1:] if path.anchor else path.parts
        if any(":" in part or PureWindowsPath(part).is_reserved() for part in parts):
            raise ValueError("device paths and alternate streams are outside the command scope")
    return (base / path).resolve()


def card_action_paths(
    action: CardAction, policy: EffectivePolicy, *, target_path: str | None = None
) -> tuple[Path, ...] | None:
    """Return affected canonical paths when the direct command fits the execution ceiling."""
    if (
        not _single_command(action.argv)
        or not action.paths
        or "shell" not in policy.tool_classes
        or (action.network == "on" and policy.network != "on")
    ):
        return None
    try:
        cwd = _canonical_path(action.cwd, policy.folder)
        paths = tuple(_canonical_path(path, cwd) for path in action.paths)
        if target_path is not None and _canonical_path(Path(target_path), cwd) not in paths:
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    if not all(path.is_relative_to(policy.folder) for path in (cwd, *paths)):
        return None
    return paths


def card_matches(
    card: ApprovalCard,
    action: CardAction,
    paths: tuple[Path, ...],
    *,
    action_class: str,
    tier: ApprovalTier,
    task_id: str,
) -> bool:
    matcher = card.matcher
    command_len = len(matcher.command)
    return (
        card.action_class == action_class
        and card.tier == tier
        and (card.task_id is None or card.task_id == task_id)
        and (card.project_id is None or card.project_id == action.project_id)
        and action.network == matcher.network
        and (matcher.branch is None or matcher.branch == action.branch)
        and action.argv[:command_len] == matcher.command
        and (matcher.arguments is None or action.argv[command_len:] == matcher.arguments)
        and (matcher.path_prefix is None or all(path.is_relative_to(matcher.path_prefix) for path in paths))
    )
