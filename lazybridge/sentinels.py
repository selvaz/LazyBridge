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
    """Use the Envelope produced by a specific parallel branch.

    Alias of :class:`_FromStep` — forwards a single branch's envelope.
    For aggregating ALL siblings in a parallel band, use
    :class:`_FromParallelAll` (``from_parallel_all("name")``).
    """

    name: str


@dataclass(frozen=True)
class _FromParallelAll:
    """Aggregate every consecutive parallel sibling starting at ``name``.

    The runtime walks the contiguous block of ``parallel=True`` steps that
    begins at the named step (in declared order) and returns ONE envelope
    that carries all of them:

    - ``task``     : labelled-text join
                     (``"[branch_a]\\n<text>\\n\\n[branch_b]\\n<text>..."``)
                     so ordinary LLM steps consume it without changes.
    - ``payload``  : the same labelled-text join string as ``task`` — NOT
                     a ``list[Envelope]``.  Use ``from_parallel("name")``
                     to access an individual branch's typed payload.
    - ``metadata`` : summed input/output tokens and cost across branches.
    - ``error``    : first non-None branch error if any
                     (short-circuit semantics — caller can detect failure).

    Compile-time check: the named step must (a) exist, (b) come earlier in
    the plan, and (c) itself be ``parallel=True`` (otherwise the "band" is
    a single step and the result would be indistinguishable from ``from_step``).
    """

    name: str


# Public singletons / factories
from_prev = _FromPrev()
from_start = _FromStart()


def from_step(name: str) -> _FromStep:
    return _FromStep(name=name)


def from_parallel(name: str) -> _FromParallel:
    return _FromParallel(name=name)


def from_parallel_all(name: str) -> _FromParallelAll:
    """Aggregate every consecutive parallel sibling starting at ``name``."""
    return _FromParallelAll(name=name)


Sentinel = _FromPrev | _FromStart | _FromStep | _FromParallel | _FromParallelAll
