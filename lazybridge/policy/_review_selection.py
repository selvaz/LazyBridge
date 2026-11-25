"""Select the coding engine for an independent review."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, NamedTuple

from .errors import PolicyError

ReviewMode = Literal["opposite", "opposite-or-fresh"]
ReviewIndependence = Literal["opposite-engine", "same-provider-fresh"]

_OPPOSITE_ENGINES = {"claude": "codex", "codex": "claude"}


class ReviewSelection(NamedTuple):
    """Reviewer engine and the independence required for that selection."""

    engine: str
    independence: ReviewIndependence


def _engine_name(engine: str) -> str:
    name = engine.strip().lower()
    return "claude" if name == "claude-code" else name


def opposite_engine_reviewer(writer_engine: str) -> str:
    """Return the independent reviewer engine for a supported writer."""

    try:
        return _OPPOSITE_ENGINES[_engine_name(writer_engine)]
    except (AttributeError, KeyError) as exc:
        raise PolicyError(f"unsupported writer engine: {writer_engine!r}") from exc


def select_reviewer(
    writer_engine: str,
    available_engines: Iterable[str],
    mode: ReviewMode = "opposite",
) -> ReviewSelection:
    """Prefer the opposite engine and require a fresh reviewer on fallback.

    Availability describes the configured coding engines. Same-provider
    fallback requires the writer's engine to be the only available engine.
    Raises PolicyError if the mode allows no available reviewer.
    """

    if mode not in ("opposite", "opposite-or-fresh"):
        raise PolicyError(f"unsupported review mode: {mode!r}")
    opposite = opposite_engine_reviewer(writer_engine)
    available = {_engine_name(engine) for engine in available_engines}
    if opposite in available:
        return ReviewSelection(opposite, "opposite-engine")
    writer = _engine_name(writer_engine)
    if mode == "opposite-or-fresh" and available == {writer}:
        return ReviewSelection(writer, "same-provider-fresh")
    raise PolicyError(f"opposite reviewer engine {opposite!r} is unavailable")
