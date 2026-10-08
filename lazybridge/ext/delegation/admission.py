"""Generic, policy-free admission/reservation protocol for delegation tools.

``admission_gate`` callables threaded through this package (background,
parallel, and plan delegation) return either ``None`` (no admission concept
-- always allowed, today's default) or a duck-typed decision object:

* ``allowed: bool`` -- whether the call may proceed.
* ``rejection_text() -> str`` -- a human-readable refusal, used when
  ``allowed`` is ``False``. A plain ``.reason`` attribute is accepted as a
  fallback (see :func:`rejection_text`) for a caller whose admission object
  predates this method.
* ``release() -> None`` (optionally awaitable) -- give back whatever this
  decision reserved, once the attempt it was reserved for has actually run
  to completion (success or failure). Optional: an admission object with no
  reservation to give back simply omits it.
* ``refund() -> None`` (optionally awaitable) -- give back the reservation
  AND anything else the caller's own admission considers spent on an
  attempt that never ran at all (the motivating case this was promoted
  from: a human operator's one-use approval, burned on a delegation a
  later quota check then refused, or a plan task claimed and then never
  started). Falls back to ``release()`` when absent -- a caller with
  nothing extra to give back needs only one method.

This module carries no quota, approval, or reservation POLICY of its own --
only the plumbing every delegate in this package uses to call whichever of
``release``/``refund`` fits the exit path it is on, exactly once, without
ever needing to know what either one actually gives back.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Protocol, runtime_checkable

_logger = logging.getLogger(__name__)


class _AdmissionLease:
    """Attempt-local, single settlement around an arbitrary caller decision."""

    def __init__(self, decision: Any) -> None:
        self.decision = decision
        self.allowed = getattr(decision, "allowed", True)
        self.settled = False

    async def release(self) -> None:
        if not self.settled:
            self.settled = True
            await release_admission(self.decision)

    async def refund(self) -> None:
        if not self.settled:
            self.settled = True
            await refund_admission(self.decision)


@runtime_checkable
class AdmissionDecision(Protocol):
    """The shape an ``admission_gate``'s non-``None`` return value is
    expected to satisfy. Never ``isinstance``-checked by this package
    (every helper below uses ``getattr`` with a default, exactly as the
    pre-existing ``allowed``/``reason`` convention already did) -- this
    exists for callers' own type hints, not as a runtime gate."""

    allowed: bool

    def rejection_text(self) -> str: ...

    def release(self) -> Any: ...

    def refund(self) -> Any: ...


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise pass it through.

    Lets ``release``/``refund`` be either plain sync methods (the common
    case -- most reservation stores are synchronous) or async ones, without
    this package caring which."""
    if inspect.isawaitable(value):
        return await value
    return value


def rejection_text(admission: Any) -> str:
    """Human-readable refusal text for a denied ``admission``.

    Prefers the ``rejection_text()`` method this protocol defines; falls
    back to a plain ``.reason`` attribute for the simpler admission objects
    this package's own tests and docstrings used before that method
    existed, so neither shape is a breaking change for the other.
    """
    method = getattr(admission, "rejection_text", None)
    if callable(method):
        try:
            text = method()
        except Exception:
            _logger.exception("admission.rejection_text() raised")
            text = None
        if text:
            return str(text)
    return str(getattr(admission, "reason", None) or "rejected by admission_gate")


async def release_admission(admission: Any) -> None:
    """Give back one granted admission's reservation, if there is one.

    A no-op for ``None`` (no admission concept in play), for a *denied*
    admission (``allowed`` is falsy -- nothing was ever granted to give
    back), and for an admission object exposing no ``release`` method at
    all (a minimal admission_gate that only ever answers allow/deny).
    Swallows any exception ``release()`` itself raises -- a broken
    reservation accounting must not also break the delegation cleanup path
    that is trying to call it, on every one of which this runs at most
    once.
    """
    if admission is None or not getattr(admission, "allowed", True):
        return
    release = getattr(admission, "release", None)
    if release is None:
        return
    try:
        await _maybe_await(release())
    except Exception:
        _logger.exception("admission.release() failed")


async def refund_admission(admission: Any) -> None:
    """Give back a granted admission's reservation AND anything else the
    caller's own admission considers spent, for an attempt that never ran
    at all.

    Falls back to :func:`release_admission` when the admission object
    exposes no separate ``refund`` -- a caller with nothing extra to give
    back needs no second method. Same ``None``/denied/exception-swallowing
    behaviour as :func:`release_admission` otherwise.
    """
    if admission is None or not getattr(admission, "allowed", True):
        return
    refund = getattr(admission, "refund", None)
    if refund is None:
        await release_admission(admission)
        return
    try:
        await _maybe_await(refund())
    except Exception:
        _logger.exception("admission.refund() failed")


__all__ = [
    "AdmissionDecision",
    "refund_admission",
    "rejection_text",
    "release_admission",
]
