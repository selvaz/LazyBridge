"""Sentinels — typed markers for Step input/context resolution in PlanEngine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _FromPrev:
    """Use the Envelope produced by the previous step (default)."""
    pass


@dataclass(frozen=True)
class _FromStart:
    """Use the initial task/context Envelope passed to the Plan."""
    pass


@dataclass(frozen=True)
class _FromStep:
    """Use the Envelope produced by a named step."""
    name: str


@dataclass(frozen=True)
class _FromParallel:
    """Use the Envelope produced by a specific parallel branch."""
    name: str


# Public singletons / factories
from_prev = _FromPrev()
from_start = _FromStart()


def from_step(name: str) -> _FromStep:
    return _FromStep(name=name)


def from_parallel(name: str) -> _FromParallel:
    return _FromParallel(name=name)


Sentinel = _FromPrev | _FromStart | _FromStep | _FromParallel
